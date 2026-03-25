# Self-evolving-Agent GPU 启动脚本使用说明

## 一、修改/新增文件清单

### 新增文件

| 文件路径 | 说明 |
|----------|------|
| `se_code_auto/__init__.py` | Python 包初始化文件，使目录成为可导入的包 |
| `se_code_auto/gpu_config.py` | GPU 配置模块，集中管理 GPU 分配 |
| `se_code_auto/Zero/run_with_gpus.sh` | **新的对外入口脚本**，GPU 检测/等待/分配 |
| `se_code_auto/ARCHITECTURE.md` | 项目架构分析文档 |
| `se_code_auto/README_GPU_LAUNCH.md` | 本文档 |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `se_code_auto/Zero/challenger.sh` | 从环境变量读取 GPU 配置，替换硬编码 |
| `se_code_auto/Zero/solver.sh` | 从环境变量读取 GPU 配置，替换硬编码 |
| `se_code_auto/Zero/challenger_reward.sh` | 动态启动多个 Reward Server |
| `se_code_auto/Zero/gen_query.sh` | 动态在多个 GPU 上并行生成数据 |
| `se_code_auto/reward_manager.py` | 从环境变量读取端口配置，替换硬编码 |
| `se_code_auto/reward.py` | 修复导入路径问题 |

## 二、使用方法

### 2.1 基本用法

```bash
cd /home/ycy/data1/Self-evolving-Agent/se_code_auto/Zero

# 使用 4 张 GPU
./run_with_gpus.sh 4

# 使用 8 张 GPU
./run_with_gpus.sh 8
```

### 2.2 执行流程

```
./run_with_gpus.sh 4
    │
    ├── 1. 检测当前 GPU 状态（显存、利用率）
    │
    ├── 2. 等待空闲 GPU ≥ 4（轮询检查）
    │
    ├── 3. 选择 4 张空闲 GPU（如 0,1,2,3）
    │
    ├── 4. 设置环境变量
    │   ├── SE_N_GPUS=4
    │   ├── SE_GPU_IDS=0,1,2,3
    │   ├── SE_CHALLENGER_GPUS=0,1
    │   ├── SE_REWARD_GPUS=2,3
    │   ├── SE_SOLVER_GPUS=0,1,2,3
    │   └── SE_REWARD_PORTS=5000,5001
    │
    └── 5. 调用 main.sh 启动训练
```

### 2.3 GPU 分配规则

| n_gpus | Challenger | Reward Server | Solver |
|--------|------------|---------------|--------|
| 4 | 前 2 张 (0,1) | 后 2 张 (2,3) | 全部 4 张 |
| 8 | 前 4 张 (0,1,2,3) | 后 4 张 (4,5,6,7) | 全部 8 张 |

## 三、环境变量说明

启动脚本会设置以下环境变量，供各训练脚本读取：

### 3.1 GPU 相关环境变量

| 环境变量 | 说明 | 示例值 (n_gpus=4) |
|----------|------|-------------------|
| `SE_N_GPUS` | 总 GPU 数量 | 4 |
| `SE_GPU_IDS` | 所有选中的 GPU ID | 0,1,2,3 |
| `SE_CHALLENGER_GPUS` | Challenger 训练 GPU | 0,1 |
| `SE_N_CHALLENGER_GPUS` | Challenger GPU 数量 | 2 |
| `SE_REWARD_GPUS` | Reward Server GPU | 2,3 |
| `SE_N_REWARD_GPUS` | Reward GPU 数量 | 2 |
| `SE_SOLVER_GPUS` | Solver 训练 GPU | 0,1,2,3 |
| `SE_N_SOLVER_GPUS` | Solver GPU 数量 | 4 |
| `SE_GEN_QUERY_GPUS` | 数据生成 GPU | 0,1,2,3 |
| `SE_REWARD_PORTS` | Reward Server 端口 | 5000,5001 |
| `SE_REWARD_BASE_PORT` | 基础端口 | 5000 |

### 3.2 路径相关环境变量

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `SE_BASE_DIR` | 基础目录 | /home/ycy/data1 |
| `SE_PROJECT_NAME` | 项目名称 | Self-evolving-Agent |
| `SE_CODE_MODULE` | 代码模块名 | se_code_auto |
| `SE_WORKING_DIR` | 工作目录 | ${SE_BASE_DIR}/${SE_PROJECT_NAME} |
| `SE_MODEL_DIR` | 模型目录 | ${SE_BASE_DIR}/models |
| `SE_DATA_DIR` | 数据目录 | ${SE_BASE_DIR}/data |
| `SE_SAVED_RESULTS_DIR` | 结果保存目录 | ${SE_BASE_DIR}/saved_results |
| `SE_CHALLENGER_DIR` | Challenger 输出目录 | ${SE_SAVED_RESULTS_DIR}/Challenger |
| `SE_SOLVER_DIR` | Solver 输出目录 | ${SE_SAVED_RESULTS_DIR}/Solver |
| `SE_TENSORBOARD_DIR` | TensorBoard 日志目录 | ${SE_SAVED_RESULTS_DIR}/tensorboard_log |
| `SE_PROMPT_DIR` | Prompt 文件目录 | ${SE_WORKING_DIR}/${SE_CODE_MODULE} |
| `SE_BASE_MODEL_NAME` | 基础模型名称 | Qwen3-4B-Base |
| `SE_BASE_MODEL_PATH` | 基础模型路径 | ${SE_MODEL_DIR}/${SE_BASE_MODEL_NAME} |
| `SE_N_REWARD_SERVERS` | Reward Server 数量 | 2 |

## 四、配置参数

可通过环境变量自定义 GPU 检测行为：

```bash
# GPU 空闲判定阈值
export GPU_MEMORY_THRESHOLD_MB=500    # 显存占用低于此值视为空闲 (MB)
export GPU_UTIL_THRESHOLD=10          # 利用率低于此值视为空闲 (%)

# 轮询配置
export POLL_INTERVAL=30               # 检查间隔（秒）
export MAX_WAIT_HOURS=48              # 最大等待时间（小时）

# Reward Server 端口
export SE_REWARD_BASE_PORT=5000       # 基础端口

# 然后运行
./run_with_gpus.sh 4
```

## 五、训练阶段时序图

```
时间线 ──────────────────────────────────────────────────────────────►

阶段 1: Challenger 训练
┌─────────────────────────────────────────────────────────────────────┐
│ GPU 0,1: Challenger 训练 (python -m se_code_auto.main_challenger)   │
│ GPU 2,3: Reward Server (start_vllm_server.py port=5000,5001)        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
阶段 2: Query 生成
┌─────────────────────────────────────────────────────────────────────┐
│ GPU 0,1,2,3: 并行 challenger_generate_query.py                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
阶段 3: Solver 训练
┌─────────────────────────────────────────────────────────────────────┐
│ GPU 0,1,2,3: Solver 训练 (python -m se_code_auto.main_solver_dapo)  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    重复阶段 1-3 (多轮迭代)
```

## 六、关键日志输出

运行时会输出以下关键信息：

```
==============================================
  Self-evolving-Agent GPU Launcher
==============================================
时间: 2025-12-12 18:00:00

[INFO] 18:00:00 - 请求 GPU 数量: 4

当前 GPU 状态:
----------------------------------------
GPU   显存使用     显存总量     利用率    状态
----------------------------------------
0     100MB       24576MB     0%        空闲
1     100MB       24576MB     0%        空闲
2     100MB       24576MB     0%        空闲
3     100MB       24576MB     0%        空闲
4     20000MB     24576MB     85%       占用
5     20000MB     24576MB     90%       占用
----------------------------------------

[INFO] 18:00:00 - 当前空闲 GPU: 4 / 需要: 4
[INFO] 18:00:00 - GPU 资源满足要求!
[INFO] 18:00:00 - 选中 GPU: 0 1 2 3

GPU 分配配置:
==============================================
总 GPU 数量:       4
选中 GPU IDs:      0,1,2,3

Challenger GPUs:   0,1 (共 2 张)
Reward GPUs:       2,3 (共 2 张)
Solver GPUs:       0,1,2,3 (共 4 张)
Gen Query GPUs:    0,1,2,3

Reward Ports:      5000,5001
Reward Base Port:  5000
==============================================

[INFO] 18:00:00 - 启动训练...
```

## 七、故障排除

### 7.1 GPU 等待超时

如果出现 "等待超时" 错误，说明服务器长时间没有足够的空闲 GPU。

解决方案：
1. 检查其他用户是否占用 GPU
2. 增大 `MAX_WAIT_HOURS` 环境变量
3. 降低 `n_gpus` 参数值

### 7.2 端口冲突

如果 Reward Server 启动失败，可能是端口被占用。

解决方案：
```bash
# 检查端口占用
lsof -i:5000
lsof -i:5001

# 更换基础端口
export SE_REWARD_BASE_PORT=6000
./run_with_gpus.sh 4
```

### 7.3 导入错误

如果出现 `ModuleNotFoundError`，确保：

1. 工作目录正确：
```bash
cd /home/ycy/data1/Self-evolving-Agent
```

2. PYTHONPATH 包含项目根目录：
```bash
export PYTHONPATH=/home/ycy/data1/Self-evolving-Agent:$PYTHONPATH
```

## 八、与 main.sh 的关系

```
run_with_gpus.sh (新入口)
    │
    │  设置 GPU 环境变量 + 路径环境变量
    │
    └──► main.sh (原入口，读取环境变量)
            │
            ├──► challenger.sh (读取 SE_CHALLENGER_GPUS, SE_BASE_DIR 等)
            │       └──► challenger_reward.sh (读取 SE_REWARD_GPUS, SE_REWARD_PORTS, SE_CODE_MODULE)
            │
            └──► solver.sh (读取 SE_SOLVER_GPUS, SE_BASE_DIR 等)
                    └──► gen_query.sh (读取 SE_GEN_QUERY_GPUS, SE_CODE_MODULE)
```

**main.sh 核心逻辑不变**，只是改为从环境变量读取路径配置。

## 九、自定义路径配置

如果需要在不同环境运行，可以通过环境变量覆盖默认路径：

```bash
# 方式1: 在运行前设置环境变量
export SE_BASE_DIR=/data/your_path
export SE_MODEL_DIR=/models/your_models
./run_with_gpus.sh 4

# 方式2: 内联设置
SE_BASE_DIR=/data/your_path SE_MODEL_DIR=/models/your_models ./run_with_gpus.sh 4
```

## 十、修改文件完整清单

### 新增文件
- `se_code_auto/__init__.py` - Python 包初始化
- `se_code_auto/gpu_config.py` - GPU 配置模块
- `se_code_auto/Zero/run_with_gpus.sh` - 新的启动入口
- `se_code_auto/ARCHITECTURE.md` - 架构分析文档
- `se_code_auto/README_GPU_LAUNCH.md` - 本文档

### 修改文件
- `se_code_auto/Zero/main.sh` - 路径改为环境变量
- `se_code_auto/Zero/challenger.sh` - GPU/路径改为环境变量
- `se_code_auto/Zero/solver.sh` - GPU/路径改为环境变量
- `se_code_auto/Zero/challenger_reward.sh` - GPU/端口/路径改为环境变量
- `se_code_auto/Zero/gen_query.sh` - GPU/路径改为环境变量
- `se_code_auto/reward_manager.py` - 端口配置改为环境变量
- `se_code_auto/reward.py` - 修复导入路径
- `se_code_auto/Challenger_dataset.py` - 路径改为环境变量
