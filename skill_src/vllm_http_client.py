"""
vLLM HTTP 客户端：通过 OpenAI-compatible API 调用远程 vLLM server。

用于适配由 vllm serve 启动的 vLLM server，替代直接 Python vLLM 初始化。
支持多 server 负载均衡和错误重试。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import httpx

logger = logging.getLogger(__name__)


class VLLMHTTPClient:
    """
    通过 OpenAI-compatible API 调用远程 vLLM server。
    
    支持：
    - 多 server 轮询负载均衡
    - 错误重试机制
    - 异步和同步调用
    - 响应格式转换
    """

    def __init__(
        self,
        server_urls: List[str],
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        served_model_name: Optional[str] = None,
        max_concurrent: int = 0,
    ):
        """
        初始化 HTTP 客户端。
        
        Args:
            server_urls: vLLM server URLs 列表，如 ["http://127.0.0.1:8760", ...]
            timeout: HTTP 请求超时时间（秒），主要作为 **读超时**（排队+生成整段响应）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            served_model_name: /v1/completions 请求体中的 model 名；默认读环境变量
                VLLM_SERVED_MODEL_NAME 或 SERVED_MODEL_NAME，再退回 "default"（须与 vllm serve --served-model-name 一致）
            max_concurrent: 同时对 vLLM 发起的 /v1/completions 请求上限；0 表示不限制。
                多分片并行时每进程各开一批并发，极易把单卡队列撑爆导致 ReadTimeout，建议 16–64。
        """
        if not server_urls:
            raise ValueError("server_urls 不能为空")
        
        self.server_urls = server_urls
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max(0, int(max_concurrent))
        self._round_robin_idx = 0
        self._client = None
        if served_model_name and str(served_model_name).strip():
            self.served_model_name = str(served_model_name).strip()
        else:
            self.served_model_name = (
                os.environ.get("VLLM_SERVED_MODEL_NAME", "").strip()
                or os.environ.get("SERVED_MODEL_NAME", "").strip()
                or "default"
            )

    def _get_next_server(self) -> str:
        """轮询获取下一个 server URL"""
        url = self.server_urls[self._round_robin_idx]
        self._round_robin_idx = (self._round_robin_idx + 1) % len(self.server_urls)
        return url

    async def health_check(self) -> Dict[str, Any]:
        """检查 server 健康状态"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in self.server_urls:
                try:
                    resp = await client.get(f"{url}/health")
                    if resp.status_code == 200:
                        return resp.json()
                except Exception as e:
                    logger.warning(f"Health check failed for {url}: {e}")
        return {"ok": False, "error": "All servers unreachable"}

    async def generate_async(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        request_timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        异步调用 vLLM server 生成补全。
        
        Args:
            prompts: prompt 字符串列表
            sampling_params: 采样参数（max_tokens, temperature, top_p, top_k, n 等）
            request_timeout: 单个请求超时时间，默认使用 self.timeout
            
        Returns:
            返回格式与原 vllm.LLM.generate() 兼容的对象列表
        """
        timeout = request_timeout or self.timeout

        sem: Optional[asyncio.Semaphore] = None
        if self.max_concurrent > 0:
            sem = asyncio.Semaphore(self.max_concurrent)

        async def _bounded(p: str) -> Any:
            if sem is not None:
                async with sem:
                    return await self._generate_single_async(
                        p, sampling_params, timeout
                    )
            return await self._generate_single_async(p, sampling_params, timeout)

        tasks = [_bounded(p) for p in prompts]
        completions = await asyncio.gather(*tasks)
        return completions

    async def _generate_single_async(
        self,
        prompt: str,
        sampling_params: Dict[str, Any],
        timeout: float,
    ) -> Any:
        """
        为单个 prompt 异步生成补全。
        
        Returns:
            模拟 vllm.RequestOutput 的对象
        """
        last_detail = ""
        # 读超时覆盖「排队等待 + 生成整段 body」；与 connect/pool 分离避免误杀长推理
        hx_timeout = httpx.Timeout(
            connect=60.0,
            read=timeout,
            write=120.0,
            pool=60.0,
        )
        for attempt in range(self.max_retries):
            url = self._get_next_server()
            try:
                async with httpx.AsyncClient(timeout=hx_timeout) as client:
                    # OpenAI /v1/completions 的 stop 只能是字符串；整数列表须用 vLLM 扩展字段 stop_token_ids
                    request_body: Dict[str, Any] = {
                        "model": self.served_model_name,
                        "prompt": prompt,
                        "max_tokens": sampling_params.get("max_tokens", 4096),
                        "temperature": sampling_params.get("temperature", 1.0),
                        "top_p": sampling_params.get("top_p", 1.0),
                        "n": sampling_params.get("n", 1),
                    }
                    top_k = sampling_params.get("top_k", -1)
                    if top_k is not None and int(top_k) >= 0:
                        request_body["top_k"] = int(top_k)
                    stop = sampling_params.get("stop")
                    extra_ids = sampling_params.get("stop_token_ids")
                    if extra_ids:
                        request_body["stop_token_ids"] = list(extra_ids)
                    elif (
                        isinstance(stop, list)
                        and stop
                        and all(isinstance(x, int) for x in stop)
                    ):
                        request_body["stop_token_ids"] = stop
                    elif stop:
                        request_body["stop"] = stop

                    logger.debug(f"POST {url}/v1/completions with {len(prompt)} chars")
                    resp = await client.post(
                        f"{url}/v1/completions",
                        json=request_body,
                        timeout=hx_timeout,
                    )

                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception as json_err:
                            # JSON 解析失败（包括空 body）
                            raw_text = (resp.text or "")[:500]
                            raise RuntimeError(
                                f"JSON parse error from {url}: {json_err}. Raw text: {raw_text!r}"
                            ) from json_err
                        if data is None:
                            raise RuntimeError(
                                f"Empty JSON body from {url} (status 200 but body is None)"
                            )
                        return _parse_openai_completion_response(data)
                    snippet = (resp.text or "")[:800]
                    last_detail = f"HTTP {resp.status_code} from {url}: {snippet}"
                    logger.warning(
                        f"Server {url} returned {resp.status_code}: {snippet[:200]}"
                    )

            except asyncio.TimeoutError:
                last_detail = f"timeout waiting for {url} (attempt {attempt + 1})"
                logger.warning(last_detail)
            except Exception as e:
                last_detail = f"{type(e).__name__} from {url}: {e}"
                logger.warning(
                    f"Request failed for {url} (attempt {attempt + 1}): {e}"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)

        raise RuntimeError(
            f"Failed to generate after {self.max_retries} attempts for prompt; last_error={last_detail}"
        )

    def generate_sync(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        request_timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        同步调用 vLLM server 生成补全（使用 asyncio.run 包装异步调用）。
        """
        return asyncio.run(
            self.generate_async(prompts, sampling_params, request_timeout)
        )


class MockRequestOutput:
    """模拟 vllm.RequestOutput 对象，兼容原有的处理流程"""
    
    def __init__(self, outputs: List[str]):
        """
        Args:
            outputs: 生成的文本列表
        """
        self.outputs = [MockOutput(text) for text in outputs]


class MockOutput:
    """模拟 vllm.RequestOutput.Output 对象"""
    
    def __init__(self, text: str):
        self.text = text


def _parse_openai_completion_response(response_data: Dict[str, Any]) -> MockRequestOutput:
    """
    将 OpenAI API 的 /v1/completions 响应转换为 vLLM RequestOutput 格式。

    OpenAI API 响应格式:
    {
        "id": "...",
        "object": "text_completion",
        "created": ...,
        "model": "...",
        "choices": [
            {"text": "...", "index": 0, "logprobs": null, "finish_reason": "..."},
            ...
        ],
        "usage": {...}
    }
    """
    if response_data is None:
        raise ValueError("Response JSON is None (empty body or parse error)")
    if not isinstance(response_data, dict):
        raise ValueError(f"Response JSON is not a dict: {type(response_data)}")

    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty choices in response: {response_data.keys()}")

    # 按 index 排序以保证顺序
    sorted_choices = sorted(choices, key=lambda x: x.get("index", 0))
    texts = [choice.get("text", "") for choice in sorted_choices]

    return MockRequestOutput(texts)
