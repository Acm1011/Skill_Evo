#!/usr/bin/env bash

########################################
# test_vllm_http_integration.sh
# 测试 vLLM HTTP 化适配系统的基本功能
########################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== vLLM HTTP 化适配 - 功能测试 =========="
echo ""

# 检查必需的文件
echo "【检查】必需的文件是否存在..."
files_to_check=(
  "skill_src/vllm_http_client.py"
  "skill_src/solver_offline_rollout_http_client.py"
  "skill_src/start_vllm_http_servers.sh"
  "VLLM_HTTP_INTEGRATION.md"
)

all_exist=true
for file in "${files_to_check[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✓ $file"
  else
    echo "  ✗ $file (缺失!)"
    all_exist=false
  fi
done

if [ "$all_exist" = false ]; then
  echo ""
  echo "错误：部分必需文件缺失，请检查"
  exit 1
fi

echo ""
echo "【检查】Python 模块导入..."

# 检查 vllm_http_client 是否可导入
if python3 -c "import sys; sys.path.insert(0, '.'); from skill_src.vllm_http_client import VLLMHTTPClient; print('  ✓ VLLMHTTPClient 导入成功')" 2>/dev/null; then
  echo "  ✓ VLLMHTTPClient 导入成功"
else
  echo "  ✗ VLLMHTTPClient 导入失败"
  exit 1
fi

# 检查 solver_offline_rollout_http_client 是否可导入
if python3 -c "import sys; sys.path.insert(0, '.'); from skill_src.solver_offline_rollout_http_client import create_app; print('  ✓ create_app 导入成功')" 2>/dev/null; then
  echo "  ✓ create_app 导入成功"
else
  echo "  ✗ create_app 导入失败"
  exit 1
fi

echo ""
echo "【检查】bash 脚本格式..."

# 检查 start_vllm_http_servers.sh 的 bash 语法
if bash -n "skill_src/start_vllm_http_servers.sh" 2>/dev/null; then
  echo "  ✓ start_vllm_http_servers.sh 语法正确"
else
  echo "  ✗ start_vllm_http_servers.sh 语法错误"
  exit 1
fi

echo ""
echo "【检查】VLLMHTTPClient 类的基本功能..."

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from skill_src.vllm_http_client import VLLMHTTPClient, MockRequestOutput, _parse_openai_completion_response
import json

# 测试 MockRequestOutput
print("  ✓ MockRequestOutput 类可用")

# 测试响应解析
sample_response = {
    "id": "test-1",
    "object": "text_completion",
    "created": 12345,
    "model": "test",
    "choices": [
        {"text": "Hello", "index": 0, "finish_reason": "stop"},
        {"text": "World", "index": 1, "finish_reason": "stop"},
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20}
}
result = _parse_openai_completion_response(sample_response)
assert len(result.outputs) == 2
assert result.outputs[0].text == "Hello"
assert result.outputs[1].text == "World"
print("  ✓ OpenAI 响应解析正确")

# 测试客户端初始化
try:
    client = VLLMHTTPClient(
        server_urls=["http://127.0.0.1:8760"],
        timeout=300.0
    )
    print("  ✓ VLLMHTTPClient 初始化成功")
except Exception as e:
    print(f"  ✗ VLLMHTTPClient 初始化失败: {e}")
    sys.exit(1)

EOF

echo ""
echo "【检查】FastAPI 应用创建..."

python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import os

from skill_src.solver_offline_rollout_http_client import create_app

try:
    app = create_app(
        vllm_server_urls=["http://127.0.0.1:8760"],
        model_path="/tmp/dummy_model",
        timeout=300.0
    )
    print("  ✓ FastAPI 应用创建成功")
    print("  ✓ 应用路由列表:")
    for route in app.routes:
        if hasattr(route, 'path'):
            print(f"    - {route.path} ({route.methods if hasattr(route, 'methods') else 'N/A'})")
except Exception as e:
    print(f"  ✗ FastAPI 应用创建失败: {e}")
    sys.exit(1)

EOF

echo ""
echo "========== 测试通过 =========="
echo ""
echo "所有基本功能检查已完成。系统已准备就绪。"
echo ""
echo "下一步："
echo "  1. 查看集成文档：VLLM_HTTP_INTEGRATION.md"
echo "  2. 运行快速开始脚本：bash QUICKSTART_VLLM_HTTP.sh"
echo "  3. 开始集成到 Synthesizer.sh"
echo ""
