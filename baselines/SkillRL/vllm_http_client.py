"""
vLLM HTTP 客户端：OpenAI-compatible /v1/completions（自包含副本，不依赖 skill_src）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class VLLMHTTPClient:
    def __init__(
        self,
        server_urls: List[str],
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        served_model_name: Optional[str] = None,
        max_concurrent: int = 0,
    ):
        if not server_urls:
            raise ValueError("server_urls 不能为空")
        self.server_urls = server_urls
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max(0, int(max_concurrent))
        self._round_robin_idx = 0
        if served_model_name and str(served_model_name).strip():
            self.served_model_name = str(served_model_name).strip()
        else:
            self.served_model_name = (
                os.environ.get("VLLM_SERVED_MODEL_NAME", "").strip()
                or os.environ.get("SERVED_MODEL_NAME", "").strip()
                or "default"
            )

    def _get_next_server(self) -> str:
        url = self.server_urls[self._round_robin_idx]
        self._round_robin_idx = (self._round_robin_idx + 1) % len(self.server_urls)
        return url

    async def generate_async(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        request_timeout: Optional[float] = None,
    ) -> List[Any]:
        timeout = request_timeout or self.timeout
        sem: Optional[asyncio.Semaphore] = None
        if self.max_concurrent > 0:
            sem = asyncio.Semaphore(self.max_concurrent)

        async def _bounded(p: str) -> Any:
            if sem is not None:
                async with sem:
                    return await self._generate_single_async(p, sampling_params, timeout)
            return await self._generate_single_async(p, sampling_params, timeout)

        tasks = [_bounded(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _generate_single_async(
        self,
        prompt: str,
        sampling_params: Dict[str, Any],
        timeout: float,
    ) -> Any:
        last_detail = ""
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

                    resp = await client.post(
                        f"{url}/v1/completions",
                        json=request_body,
                        timeout=hx_timeout,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data is None:
                            raise RuntimeError(f"Empty JSON from {url}")
                        return _parse_openai_completion_response(data)
                    snippet = (resp.text or "")[:800]
                    last_detail = f"HTTP {resp.status_code} from {url}: {snippet}"
                    logger.warning("Server %s: %s", url, snippet[:200])
            except asyncio.TimeoutError:
                last_detail = f"timeout waiting for {url} (attempt {attempt + 1})"
                logger.warning(last_detail)
            except Exception as e:
                last_detail = f"{type(e).__name__} from {url}: {e}"
                logger.warning("Request failed %s: %s", url, e)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
        raise RuntimeError(
            f"Failed after {self.max_retries} attempts; last_error={last_detail}"
        )

    def generate_sync(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        request_timeout: Optional[float] = None,
    ) -> List[Any]:
        return asyncio.run(
            self.generate_async(prompts, sampling_params, request_timeout)
        )


class MockRequestOutput:
    def __init__(self, outputs: List[str]):
        self.outputs = [MockOutput(text) for text in outputs]


class MockOutput:
    def __init__(self, text: str):
        self.text = text


def _parse_openai_completion_response(response_data: Dict[str, Any]) -> MockRequestOutput:
    if not isinstance(response_data, dict):
        raise ValueError(f"Response JSON is not a dict: {type(response_data)}")
    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty choices: {response_data.keys()}")
    sorted_choices = sorted(choices, key=lambda x: x.get("index", 0))
    texts = [choice.get("text", "") for choice in sorted_choices]
    return MockRequestOutput(texts)
