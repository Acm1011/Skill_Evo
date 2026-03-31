"""
在单卡上常驻 vLLM，通过 HTTP 接收 rollout 任务（供 solver_offline_driver 多机/多卡负载均衡）。

使用 ``AsyncLLMEngine``：多条 HTTP 可并发进入 FastAPI，内部对多条 prompt 并发 ``generate``，
由 vLLM 队列做 continuous batching；**HTTP 响应 JSON 结构与各 ``results`` 条目的字段**与原先
``LLM + run_rollout`` 版本一致（``ok``、``results``、``output_path``、``stats`` 等）。

每条样本需含 ``prompt``（str 或 messages 列表）、``question`` 与 ``gt`` /
``reward_model.ground_truth``，与 ``solver_offline_rollout.run_rollout`` 约定一致。

启动（需设置模型路径环境变量）：
  export ROLLOUT_SERVER_MODEL=/path/to/Qwen3-4B-Base
  CUDA_VISIBLE_DEVICES=0 uvicorn skill_src.solver_offline_rollout_server:app --host 0.0.0.0 --port 8760

或：
  python -m skill_src.solver_offline_rollout_server --model /path/to/model --port 8760 --gpu 0
"""
from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from skill_src.solver_offline_driver import load_records_from_files
from skill_src.solver_offline_rollout import (
    is_qwen3_post_trained,
    run_rollout_async,
)
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

logger = logging.getLogger(__name__)


def _configure_rollout_stats_logging() -> None:
    """Uvicorn 默认不把应用 logger 的 INFO 打到 stderr；保证吞吐日志进 server 的 log 文件。"""
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [rollout_server] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.propagate = False


_server_state: Dict[str, Any] = {}


class RolloutJob(BaseModel):
    """与 ``solver_offline_driver`` POST 体对齐；未使用的字段忽略。"""

    model_config = ConfigDict(extra="ignore")

    data_file: str = Field(default="", description="shard jsonl；与 data_records 二选一")
    data_records: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="内存样本；每条需 prompt、question、gt（或 reward_model.ground_truth）",
    )
    num_questions: Optional[int] = Field(
        default=None,
        description="使用的条数；默认取全部。不得大于实际样本数。",
    )
    suffix: str = Field(default="job", description="run_rollout 内 vLLM seed 推导")
    storage_path: str = Field(default="", description="兼容旧客户端，忽略")
    skill_type: str = "skill_generation_v1"
    rollout_n: int = 10
    max_tokens: int = 4096
    top_k: int = 50
    top_p: float = 0.95
    gpu_utilization: float = 0.95
    temperature: float = 1.0
    num_random_questions: int = 10
    seed: Optional[int] = Field(None, description="可选；不传则用 suffix 推导")


def _build_args(payload: RolloutJob, model_path: str) -> SimpleNamespace:
    """仅包含 ``run_rollout`` / ``run_rollout_async`` 实际读取的字段。"""
    return SimpleNamespace(
        model=model_path,
        rollout_n=payload.rollout_n,
        max_tokens=payload.max_tokens,
        top_k=payload.top_k,
        top_p=payload.top_p,
        gpu_utilization=payload.gpu_utilization,
        temperature=payload.temperature,
        suffix=payload.suffix,
        seed=payload.seed,
    )


def _resolve_records(job: RolloutJob) -> List[Dict[str, Any]]:
    use_memory = job.data_records is not None and len(job.data_records) > 0
    if use_memory:
        records = list(job.data_records)
    else:
        if not job.data_file or not os.path.isfile(job.data_file):
            raise HTTPException(
                status_code=400,
                detail="需提供非空 data_records 或有效的 data_file 路径",
            )
        records = load_records_from_files([job.data_file])

    n = len(records)
    if n == 0:
        raise HTTPException(status_code=400, detail="样本列表为空")

    n_take = job.num_questions if job.num_questions is not None else n
    if n_take <= 0:
        raise HTTPException(status_code=400, detail="num_questions 必须为正")
    if n < n_take:
        raise HTTPException(
            status_code=400,
            detail=f"样本数 {n} < num_questions {n_take}",
        )
    return records[:n_take]


def create_app(model: str, gpu_utilization: float = 0.95) -> FastAPI:
    _configure_rollout_stats_logging()
    app = FastAPI(title="solver_offline_rollout_server", version="1.0")

    @app.on_event("startup")
    def _load_model() -> None:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        qwen3 = is_qwen3_post_trained(model)
        seed = abs(hash(os.environ.get("SERVER_SEED", "0"))) % (2**31)
        engine_args = AsyncEngineArgs(
            model=model,
            tokenizer=model,
            gpu_memory_utilization=gpu_utilization,
            seed=seed,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        _server_state["engine"] = engine
        _server_state["tokenizer"] = tokenizer
        _server_state["qwen3_post_trained"] = qwen3
        _server_state["model_path"] = model

    @app.get("/health")
    def health() -> Dict[str, Any]:
        ok = _server_state.get("engine") is not None
        return {"ok": ok, "model": _server_state.get("model_path", "")}

    @app.post("/rollout")
    async def rollout_job(job: RolloutJob) -> Dict[str, Any]:
        records = _resolve_records(job)

        engine = _server_state.get("engine")
        tokenizer = _server_state.get("tokenizer")
        qwen3 = _server_state.get("qwen3_post_trained")
        if engine is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        args = _build_args(job, model_path=_server_state["model_path"])
        n_prompts = len(records)
        n_comp = n_prompts * int(job.rollout_n)
        t0 = time.perf_counter()
        try:
            results = await run_rollout_async(
                args,
                engine,
                tokenizer,
                qwen3,
                records,
            )
            elapsed = time.perf_counter() - t0
            pps = n_prompts / elapsed if elapsed > 0 else 0.0
            cps = n_comp / elapsed if elapsed > 0 else 0.0
            logger.info(
                "rollout done: n_prompts=%d rollout_n=%d wall_s=%.3f "
                "prompts/s=%.3f completions/s=%.3f suffix=%s",
                n_prompts,
                job.rollout_n,
                elapsed,
                pps,
                cps,
                job.suffix,
            )
            return {
                "ok": True,
                "results": results,
                "output_path": None,
                "stats": {
                    "wall_time_sec": round(elapsed, 6),
                    "n_prompts": n_prompts,
                    "rollout_n": job.rollout_n,
                    "n_completions": n_comp,
                    "prompts_per_sec": round(pps, 6),
                    "completions_per_sec": round(cps, 6),
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def _app_from_env() -> FastAPI:
    model = os.environ.get("ROLLOUT_SERVER_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "请设置环境变量 ROLLOUT_SERVER_MODEL 为模型路径，或使用 "
            "python -m skill_src.solver_offline_rollout_server --model ..."
        )
    util = float(os.environ.get("ROLLOUT_SERVER_GPU_UTIL", "0.95"))
    return create_app(model, gpu_utilization=util)


def _stub_app() -> FastAPI:
    stub = FastAPI()

    @stub.get("/health")
    def _h() -> Dict[str, Any]:
        return {
            "ok": False,
            "detail": "未设置 ROLLOUT_SERVER_MODEL；启动前 export ROLLOUT_SERVER_MODEL=/path/to/model",
        }

    return stub


try:
    app = _app_from_env()
except RuntimeError:
    app = _stub_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    _configure_rollout_stats_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8760)
    p.add_argument("--gpu-utilization", type=float, default=0.95)
    cli = p.parse_args()
    os.environ["ROLLOUT_SERVER_MODEL"] = cli.model
    os.environ["ROLLOUT_SERVER_GPU_UTIL"] = str(cli.gpu_utilization)
    app_main = create_app(cli.model, gpu_utilization=cli.gpu_utilization)
    uvicorn.run(app_main, host=cli.host, port=cli.port)
