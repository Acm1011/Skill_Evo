"""
solver_offline_rollout HTTP 客户端版本：通过 HTTP 调用远程 vLLM server。

这是 solver_offline_rollout_server.py 的替代版本，使用 vllm_http_client.py 
替代直接的 vLLM 初始化。保持相同的 FastAPI 接口和请求/响应格式。

适用场景：
- vLLM server 由 vllm serve CLI 启动（通过 start_vllm_http_servers.sh）
- 支持多 server 负载均衡
- 完全兼容现有的 solver_offline_driver 调用
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from skill_src.solver_offline_driver import load_records_from_files
from skill_src.solver_offline_rollout import (
    _gt_from_row,
    _prompt_strings,
    build_rollout_results,
    is_qwen3_post_trained,
)
from skill_src.vllm_http_client import VLLMHTTPClient
from transformers import AutoTokenizer
import vllm

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """配置日志输出到 stderr"""
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [rollout_http_client] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(h)
    logger.propagate = False


class RolloutJob(BaseModel):
    """与原 solver_offline_rollout_server 完全相同的请求格式"""

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
    suffix: str = Field(default="job", description="用于日志 / 统计")
    storage_path: str = Field(default="", description="兼容旧客户端，忽略")
    skill_type: str = "skill_generation_v1"
    rollout_n: int = 10
    max_tokens: int = 4096
    top_k: int = 50
    top_p: float = 0.95
    gpu_utilization: float = 0.95
    temperature: float = 1.0
    num_random_questions: int = 10
    seed: Optional[int] = Field(None, description="可选")


def _resolve_records(job: RolloutJob) -> List[Dict[str, Any]]:
    """解析请求中的数据源"""
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


def create_app(
    vllm_server_urls: List[str],
    model_path: str,
    timeout: float = 3600.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    served_model_name: Optional[str] = None,
    max_concurrent: int = 32,
) -> FastAPI:
    """
    创建 FastAPI 应用。
    
    Args:
        vllm_server_urls: vLLM HTTP server URLs 列表
        model_path: 模型路径（用于 tokenizer 初始化）
        timeout: HTTP 请求超时时间
        max_retries: 重试次数
        retry_delay: 重试间隔（秒）
        served_model_name: 传给 vLLM OpenAI API 的 model 名；None 时由 VLLMHTTPClient 从环境变量解析
        max_concurrent: 同时对 vLLM 的 completions 并发上限；0 为不限制（易触发 ReadTimeout）
    """
    _configure_logging()
    app = FastAPI(title="solver_offline_rollout_http_client", version="1.0")

    # 应用级状态
    _state: Dict[str, Any] = {}

    @app.on_event("startup")
    def _init() -> None:
        """初始化 tokenizer 和 HTTP client"""
        try:
            logger.info(f"初始化 tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            qwen3 = is_qwen3_post_trained(model_path)
            logger.info(f"Qwen3 post-trained: {qwen3}")

            client = VLLMHTTPClient(
                server_urls=vllm_server_urls,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                served_model_name=served_model_name,
                max_concurrent=max_concurrent,
            )
            logger.info(
                "VLLMHTTPClient 初始化完成: servers=%d timeout=%.1fs max_concurrent=%d",
                len(vllm_server_urls),
                timeout,
                max_concurrent,
            )

            _state["tokenizer"] = tokenizer
            _state["qwen3_post_trained"] = qwen3
            _state["client"] = client
            _state["model_path"] = model_path
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """健康检查"""
        ok = _state.get("client") is not None
        return {
            "ok": ok,
            "model": _state.get("model_path", ""),
            "type": "http_client",
            "servers": len(vllm_server_urls),
        }

    @app.post("/rollout")
    async def rollout_job(job: RolloutJob) -> Dict[str, Any]:
        """
        处理 rollout 请求（与原 solver_offline_rollout_server 接口完全相同）。
        
        返回格式：
        {
            "ok": bool,
            "results": [
                {
                    "question": str,
                    "responses": List[str],
                    "answers": List[str],
                    "gt": Any,
                    "is_right": List[bool],
                    "acc": float,
                },
                ...
            ],
            "output_path": None,
            "stats": {
                "wall_time_sec": float,
                "n_prompts": int,
                "rollout_n": int,
                "n_completions": int,
                "prompts_per_sec": float,
                "completions_per_sec": float,
            }
        }
        """
        records = _resolve_records(job)
        
        client = _state.get("client")
        tokenizer = _state.get("tokenizer")
        qwen3 = _state.get("qwen3_post_trained")
        
        if client is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="client not initialized")

        n_prompts = len(records)
        n_comp = n_prompts * int(job.rollout_n)
        t0 = time.perf_counter()
        
        try:
            # 获取 prompt 文本
            prompt_texts = _prompt_strings(records, tokenizer, qwen3)
            
            # 构造采样参数
            sampling_params = {
                "max_tokens": job.max_tokens,
                "temperature": job.temperature,
                "top_p": job.top_p,
                "top_k": job.top_k,
                "n": job.rollout_n,
                # vLLM HTTP：用 stop_token_ids，勿把 token id 放在 OpenAI 的 stop 字段（会 400）
                "stop_token_ids": [tokenizer.eos_token_id]
                if tokenizer.eos_token_id is not None
                else [],
            }
            
            logger.info(
                f"开始 rollout: n_prompts={n_prompts} rollout_n={job.rollout_n} "
                f"suffix={job.suffix}"
            )
            
            # 通过 HTTP client 调用 vLLM server
            completions = await client.generate_async(
                prompt_texts,
                sampling_params,
                request_timeout=timeout,
            )
            
            # 构造结果
            results = build_rollout_results(records, completions)
            
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
            logger.exception(f"rollout 失败: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def _app_from_env() -> FastAPI:
    """从环境变量读取配置"""
    vllm_urls_str = os.environ.get("VLLM_HTTP_SERVER_URLS", "").strip()
    if not vllm_urls_str:
        raise RuntimeError(
            "请设置环境变量 VLLM_HTTP_SERVER_URLS (逗号分隔的 server URLs)"
        )
    
    model = os.environ.get("ROLLOUT_SERVER_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "请设置环境变量 ROLLOUT_SERVER_MODEL 为模型路径"
        )
    
    server_urls = [url.strip() for url in vllm_urls_str.split(",") if url.strip()]
    if not server_urls:
        raise RuntimeError("VLLM_HTTP_SERVER_URLS 无效")
    
    timeout = float(os.environ.get("VLLM_HTTP_TIMEOUT", "300.0"))
    max_retries = int(os.environ.get("VLLM_HTTP_MAX_RETRIES", "3"))
    retry_delay = float(os.environ.get("VLLM_HTTP_RETRY_DELAY", "1.0"))
    max_concurrent = int(os.environ.get("VLLM_HTTP_MAX_CONCURRENT", "32"))
    served = os.environ.get("VLLM_SERVED_MODEL_NAME", "").strip() or os.environ.get(
        "SERVED_MODEL_NAME", ""
    ).strip() or None

    return create_app(
        server_urls,
        model,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        served_model_name=served,
        max_concurrent=max_concurrent,
    )


def _stub_app() -> FastAPI:
    """创建没有初始化的 stub app"""
    stub = FastAPI()

    @stub.get("/health")
    def _h() -> Dict[str, Any]:
        return {
            "ok": False,
            "detail": "未设置必需的环境变量 VLLM_HTTP_SERVER_URLS 和 ROLLOUT_SERVER_MODEL",
        }

    return stub


# 供 ``uvicorn skill_src.solver_offline_rollout_http_client:app`` 等场景使用：需预先 export
# VLLM_HTTP_SERVER_URLS / ROLLOUT_SERVER_MODEL。若用 ``python -m ... --vllm-urls ...`` 启动，导入本模块时
# 环境变量尚未设置属正常，此时为 stub；真正对外服务的是 __main__ 里 ``create_app`` + ``uvicorn.run`` 的实例。
if os.environ.get("VLLM_HTTP_SERVER_URLS", "").strip():
    try:
        app = _app_from_env()
    except RuntimeError as e:
        logger.warning("Failed to create app from env: %s", e)
        app = _stub_app()
else:
    app = _stub_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    _configure_logging()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--vllm-urls",
        type=str,
        required=True,
        help="逗号分隔的 vLLM HTTP server URLs，如 http://127.0.0.1:8760,http://127.0.0.1:8761",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径（用于 tokenizer 初始化）",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8762)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="对 vLLM 的 completions 最大并发数；0 表示读环境变量 VLLM_HTTP_MAX_CONCURRENT（默认 32）",
    )
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="vLLM 请求失败后的重试间隔（秒）；也可用环境变量 VLLM_HTTP_RETRY_DELAY",
    )
    p.add_argument(
        "--served-model-name",
        type=str,
        default="",
        help="vLLM --served-model-name 对应名称；默认同环境变量 VLLM_SERVED_MODEL_NAME / SERVED_MODEL_NAME",
    )
    
    cli = p.parse_args()
    
    server_urls = [url.strip() for url in cli.vllm_urls.split(",") if url.strip()]
    os.environ["VLLM_HTTP_SERVER_URLS"] = cli.vllm_urls
    os.environ["ROLLOUT_SERVER_MODEL"] = cli.model
    smn = (cli.served_model_name or "").strip() or None

    max_concurrent = cli.max_concurrent
    if max_concurrent <= 0:
        max_concurrent = int(os.environ.get("VLLM_HTTP_MAX_CONCURRENT", "32"))
    
    app_main = create_app(
        server_urls,
        cli.model,
        timeout=cli.timeout,
        max_retries=cli.max_retries,
        retry_delay=cli.retry_delay,
        served_model_name=smn,
        max_concurrent=max_concurrent,
    )
    uvicorn.run(app_main, host=cli.host, port=cli.port)
