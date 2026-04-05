# vLLM HTTP 化适配文档

## 概述

本文档说明如何使用新的 vLLM HTTP 化方案，将 vLLM 服务从 Python 直接初始化改为通过 `vllm serve` CLI 启动的独立 server。

这种方式具有以下优势：
- 将 vLLM server 与应用逻辑解耦
- 支持多 server 负载均衡和故障转移
- 易于扩展和管理
- 完全兼容现有的 `solver_offline_driver.py` 调用流程

## 新增文件说明

### 1. `skill_src/vllm_http_client.py`

**VLLMHTTPClient** 类：通过 OpenAI-compatible API 调用远程 vLLM server。

**主要功能：**
- 支持多 server 轮询负载均衡
- 异步/同步调用接口
- 自动重试机制
- 响应格式转换（OpenAI API → vLLM 兼容格式）

**使用示例：**
```python
from skill_src.vllm_http_client import VLLMHTTPClient
import asyncio

# 初始化客户端
client = VLLMHTTPClient(
    server_urls=["http://127.0.0.1:8760", "http://127.0.0.1:8761"],
    timeout=300.0,
    max_retries=3,
)

# 异步调用
completions = await client.generate_async(
    prompts=["Write a poem about AI"],
    sampling_params={
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "n": 10,
    }
)

# 或同步调用
completions = client.generate_sync(prompts, sampling_params)
```

### 2. `skill_src/solver_offline_rollout_http_client.py`

**FastAPI HTTP 代理层**：替代原有的 `solver_offline_rollout_server.py`。

**主要特性：**
- 完全兼容原有的 `/rollout` 接口
- 使用 `VLLMHTTPClient` 替代直接 vLLM 调用
- 相同的请求/响应格式
- 无需修改 `solver_offline_driver.py`

**启动方式：**

```bash
# 方式 1：通过命令行参数
python -m skill_src.solver_offline_rollout_http_client \
  --vllm-urls "http://127.0.0.1:8760,http://127.0.0.1:8761" \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8762

# 方式 2：通过环境变量
export VLLM_HTTP_SERVER_URLS="http://127.0.0.1:8760,http://127.0.0.1:8761"
export ROLLOUT_SERVER_MODEL="/path/to/model"
uvicorn skill_src.solver_offline_rollout_http_client:app --host 0.0.0.0 --port 8762
```

**环境变量：**
- `VLLM_HTTP_SERVER_URLS`: 逗号分隔的 vLLM server URLs（必需）
- `ROLLOUT_SERVER_MODEL`: 模型路径（必需）
- `VLLM_HTTP_TIMEOUT`: HTTP 请求超时时间，默认 300.0 秒
- `VLLM_HTTP_MAX_RETRIES`: 重试次数，默认 3

### 3. `skill_src/start_vllm_http_servers.sh`

**bash 启动脚本**：在指定的 CUDA 卡上启动多个 vLLM OpenAI API servers。

**使用方式：**

```bash
bash skill_src/start_vllm_http_servers.sh <cuda_devices> <model_path>

# 示例：在 GPU 0,1,2,3 上启动 4 个 server
bash skill_src/start_vllm_http_servers.sh 0,1,2,3 /path/to/Qwen3-4B-Base

# 带环境变量
BASE_PORT=8760 DTYPE=auto WAIT_TIMEOUT=180 \
bash skill_src/start_vllm_http_servers.sh 0,1,2,3 /path/to/model
```

**参数说明：**
- `cuda_devices`: 逗号分隔的 GPU IDs（位置参数 1）
- `model_path`: 模型路径（位置参数 2）

**可选环境变量：**
- `BASE_PORT`: 第一个 server 监听的端口，默认 8760
- `API_KEY`: vllm serve API 密钥（可选）
- `DTYPE`: 推理精度，默认 auto（可选值: auto, float32, float16, bfloat16）
- `WAIT_TIMEOUT`: 健康检查超时时间（秒），默认 180
- `TENSOR_PARALLEL_SIZE`: 张量并行大小，默认 1（分布式推理时设置）
- `SERVED_MODEL_NAME`: 对外暴露的模型名，默认 default

**启动流程：**
1. 在每个指定 GPU 上启动一个 `vllm serve` 进程
2. 第 i 个 GPU 的 server 端口为 `BASE_PORT + i`
3. 健康检查所有 server 的 `/health` 端点
4. 全部就绪后保持运行，Ctrl+C 关闭

## 集成指南

### 场景 1：替代原有 Offline Rollout 流程

**原流程：**
```bash
# skill_src/Zero/Synthesizer.sh 中
bash "${SCRIPT_DIR}/start_rollout_servers.sh" --model "${solver_model_path}" &
ROLLOUT_SERVER_PID=$!
# ... 等待 server 就绪 ...
# ... 运行 solver_offline_driver ...
```

**新流程：**
```bash
# 1. 启动 vLLM HTTP servers（纯 vLLM，不含业务逻辑）
bash "${SCRIPT_DIR}/../start_vllm_http_servers.sh" \
  "${ROLLOUT_GPU_IDS}" "${solver_model_path}" &
VLLM_SERVERS_PID=$!

# 2. 启动 HTTP 代理层（FastAPI，含业务逻辑）
export VLLM_HTTP_SERVER_URLS="http://127.0.0.1:8760,http://127.0.0.1:8761,..."
export ROLLOUT_SERVER_MODEL="${solver_model_path}"
python -m skill_src.solver_offline_rollout_http_client \
  --host 0.0.0.0 \
  --port 8762 &
ROLLOUT_PROXY_PID=$!

# 3. 等待两个服务就绪
wait_for_port 8760 300  # 等待 vLLM 就绪
wait_for_port 8762 300  # 等待代理层就绪

# 4. 现在可以继续运行 solver_offline_driver
# ... 与原流程完全相同 ...
# solver_offline_driver 会连接到 http://127.0.0.1:8762
```

### 场景 2：standalone 模式（调试）

```bash
# Terminal 1: 启动 vLLM servers
BASE_PORT=8760 bash skill_src/start_vllm_http_servers.sh 0,1 /path/to/model

# Terminal 2: 启动代理层
export VLLM_HTTP_SERVER_URLS="http://127.0.0.1:8760,http://127.0.0.1:8761"
export ROLLOUT_SERVER_MODEL="/path/to/model"
python -m skill_src.solver_offline_rollout_http_client

# Terminal 3: 运行 driver（与原流程完全相同）
python -m skill_src.solver_offline_driver run \
  --data-files /path/to/data.jsonl \
  --steps 100 \
  --batch-size 8 \
  --work-dir /tmp/rollout_work \
  --merge-output-dir /tmp/merged
```

## 配置示例

### 示例 1：4 GPU 配置

```bash
# 启动 vLLM servers（4 个 server，每个 GPU 一个）
bash skill_src/start_vllm_http_servers.sh 0,1,2,3 /path/to/Qwen3-4B-Base

# 此时 servers 监听以下端口：
# GPU 0 -> port 8760
# GPU 1 -> port 8761
# GPU 2 -> port 8762
# GPU 3 -> port 8763

# 启动代理层
export VLLM_HTTP_SERVER_URLS="http://127.0.0.1:8760,http://127.0.0.1:8761,http://127.0.0.1:8762,http://127.0.0.1:8763"
export ROLLOUT_SERVER_MODEL="/path/to/Qwen3-4B-Base"
python -m skill_src.solver_offline_rollout_http_client --port 8762
```

### 示例 2：自定义端口

```bash
# 如果想用不同的端口范围
BASE_PORT=9000 bash skill_src/start_vllm_http_servers.sh 0,1,2,3 /path/to/model

# 生成的 server URLs：
# http://127.0.0.1:9000
# http://127.0.0.1:9001
# http://127.0.0.1:9002
# http://127.0.0.1:9003

export VLLM_HTTP_SERVER_URLS="http://127.0.0.1:9000,http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003"
python -m skill_src.solver_offline_rollout_http_client --port 8762
```

## 架构流程图

```
solver_offline_driver.py
    ↓ (HTTP POST http://127.0.0.1:8762/rollout)
solver_offline_rollout_http_client.py (FastAPI, port 8762)
    ↓ (HTTP GET/POST /v1/completions)
vllm_http_client.py (负载均衡，轮询)
    ↓ (并发请求)
vLLM HTTP Servers (OpenAI-compatible API)
    ├─ http://127.0.0.1:8760 (GPU 0)
    ├─ http://127.0.0.1:8761 (GPU 1)
    ├─ http://127.0.0.1:8762 (GPU 2)
    └─ http://127.0.0.1:8763 (GPU 3)
        ↓ (vllm serve CLI)
    GPU 0, 1, 2, 3
```

## 故障排查

### 问题 1：vLLM server 启动失败

**症状：** `✗ GPU X 的 server 启动失败，端口 XXXX 在 180 秒内未就绪`

**排查步骤：**
1. 检查日志：`cat logs/vllm_gpu0_port8760.log`
2. 确认模型路径正确：`ls -la /path/to/model`
3. 确认 GPU 可用：`nvidia-smi`
4. 增加 WAIT_TIMEOUT：`WAIT_TIMEOUT=300 bash start_vllm_http_servers.sh ...`

### 问题 2：代理层连接失败

**症状：** `HTTPException(status_code=500, detail="Failed to generate after 3 attempts")`

**排查步骤：**
1. 检查 vLLM servers 是否运行：`curl http://127.0.0.1:8760/health`
2. 确认环境变量设置正确：`echo $VLLM_HTTP_SERVER_URLS`
3. 增加重试次数：`export VLLM_HTTP_MAX_RETRIES=5`
4. 增加超时时间：`export VLLM_HTTP_TIMEOUT=600.0`

### 问题 3：负载不均衡

**症状：** 某个 server 的 GPU 占用率明显高于其他 server

**解决方案：**
- 检查 rollout 任务是否均匀分配（由 `solver_offline_driver.py` 控制）
- 确保所有 vLLM servers 性能相同
- 考虑增加 server 并发处理能力（vLLM 参数调优）

## 性能调优建议

1. **调整 GPU 内存利用率**：
   ```bash
   # 在 start_vllm_http_servers.sh 中修改
   cmd+=(--gpu-memory-utilization 0.9)  # 默认 0.95
   ```

2. **启用张量并行（对于大模型）**：
   ```bash
   TENSOR_PARALLEL_SIZE=2 bash start_vllm_http_servers.sh 0,1,2,3 /path/to/model
   ```

3. **调整代理层连接池**：
   - 修改 `vllm_http_client.py` 中的 `AsyncClient` 参数

4. **启用 API key 认证**：
   ```bash
   API_KEY="my-secret-key" bash start_vllm_http_servers.sh 0,1,2,3 /path/to/model
   ```

## 完全兼容性保证

- ✅ `solver_offline_driver.py` 无需修改
- ✅ `reward_manager.py` 无需修改（仍使用原 `start_vllm_server.py`）
- ✅ `skill_src/Zero/` 中的 bash 脚本无需修改
- ✅ 所有其他业务逻辑无需修改
- ✅ 接口和数据格式 100% 兼容

## 回退方案

如需回到原有的方式，只需：
1. 继续使用原有的 `start_rollout_servers.sh`
2. 无需修改 `solver_offline_driver.py`

两种方式可以并存，通过环境变量或脚本参数切换。
