# Self-evolving-Agent 项目架构分析文档

## 一、项目整体结构

```
/home/ycy/data1/Self-evolving-Agent/
├── se_code_auto/                    # 主工作目录
│   ├── Zero/                        # 训练入口脚本目录
│   │   ├── main.sh                  # 主入口脚本
│   │   ├── challenger.sh            # Challenger 训练脚本
│   │   ├── solver.sh                # Solver 训练脚本
│   │   ├── challenger_reward.sh     # Reward Server 启动脚本
│   │   ├── gen_query.sh             # 数据生成脚本
│   │   └── process_cleanup_lib.sh   # 进程清理库
│   ├── config/                      # Hydra 配置文件
│   │   ├── challenger_trainer.yaml
│   │   └── solver_trainer.yaml
│   ├── main_challenger.py           # Challenger 训练入口
│   ├── main_solver_dapo.py          # Solver 训练入口
│   ├── Challenger_ray_trainer.py    # Challenger Ray 训练器
│   ├── Solver_dapo_ray_trainer.py   # Solver Ray 训练器
│   ├── Challenger_dataset.py        # Challenger 数据集
│   ├── reward_manager.py            # Reward 管理器
│   ├── reward.py                    # Reward 加载器
│   ├── start_vllm_server.py         # vLLM Reward Server
│   ├── challenger_generate_query.py # 查询生成脚本
│   ├── data_merge.py                # 数据合并脚本
│   ├── utils.py                     # 工具函数
│   └── evaluation/                  # 评估模块
└── verl/                            # 基础训练框架
```

## 二、主要模块职责

### 2.1 核心训练模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **Challenger 训练** | `main_challenger.py`, `Challenger_ray_trainer.py` | 生成高质量数学问题 |
| **Solver 训练** | `main_solver_dapo.py`, `Solver_dapo_ray_trainer.py` | 解决数学问题 |
| **Reward 管理** | `reward_manager.py`, `reward.py` | 计算和管理奖励 |
| **数据集** | `Challenger_dataset.py` | 生成训练提示 |

### 2.2 服务模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **Reward Server** | `start_vllm_server.py` | 为 Challenger 提供奖励计算服务 |
| **查询生成** | `challenger_generate_query.py` | 用 Challenger 模型生成问题 |
| **数据合并** | `data_merge.py` | 合并生成的数据用于 Solver 训练 |

### 2.3 辅助模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **进程管理** | `process_cleanup_lib.sh` | 进程跟踪和清理 |
| **工具函数** | `utils.py` | 格式检查、答案提取等 |
| **评估** | `evaluation/` | 模型评估 |

## 三、执行流程

### 3.1 main.sh 执行流程

```
main.sh
├── 设置环境变量和路径
├── 第 1 轮训练
│   ├── challenger.sh → 训练 Challenger (base_model → challenger_model)
│   └── solver.sh → 训练 Solver (challenger_model → solver_model)
├── 第 2~N 轮训练 (循环)
│   ├── challenger.sh → 训练 Challenger (prev_challenger + prev_solver → new_challenger)
│   └── solver.sh → 训练 Solver (new_challenger + prev_solver → new_solver)
└── 评估
    └── eval_all_run.sh → 评估所有模型
```

### 3.2 Challenger 训练流程 (challenger.sh)

```
challenger.sh <exp_name> <challenger_model_path> <solver_model_path> <steps> <reward> <penalty> <func>
├── 调用 process_cleanup_lib.sh (清理进程)
├── 调用 challenger_reward.sh (启动 vLLM Reward Server)
│   ├── GPU 6: start_vllm_server.py --port 5000
│   └── GPU 7: start_vllm_server.py --port 5001
├── 等待服务启动 (sleep 10)
├── 启动训练 (python3 -m se_code.main_challenger)
│   ├── 使用 GPU 4,5 (CUDA_VISIBLE_DEVICES=4,5)
│   ├── 使用 trainer.n_gpus_per_node=2
│   └── 使用 reward_model.reward_manager=challenger
└── 清理进程 (pkill python)
```

### 3.3 Solver 训练流程 (solver.sh)

```
solver.sh <exp_name> <challenger_model_path> <solver_model_path> <steps> <func>
├── 调用 gen_query.sh (生成训练数据)
│   ├── GPU 4,5,6,7: 并行运行 challenger_generate_query.py
│   └── 运行 data_merge.py 合并数据
├── 启动训练 (python3 -m se_code.main_solver_dapo)
│   ├── 使用 GPU 4,5,6,7 (CUDA_VISIBLE_DEVICES=4,5,6,7)
│   ├── 使用 trainer.n_gpus_per_node=4
│   └── 使用 reward_model.reward_manager=solver
└── 完成
```

## 四、关键依赖关系

### 4.1 Python 模块依赖拓扑

```
main_challenger.py
├── se_code.Challenger_dataset (ChallengerTopicDataset)
├── se_code.Challenger_ray_trainer (ChallengerRayTrainer)
├── se_code.reward (load_reward_manager)
│   ├── se_code.reward_manager (ChallengerRewardManager, SolverRewardManager)
│   └── se_code_4.reward_manager (ChallengerRewardManager - 另一个版本)
└── verl.* (基础框架)

main_solver_dapo.py
├── se_code.Solver_dapo_ray_trainer (SolverDAPOTrainer)
├── se_code.reward (load_reward_manager)
└── verl.* (基础框架)

start_vllm_server.py
├── se_code.utils (process_single_R_Zero, process_single_Rule, format_check)
└── vllm, flask (外部依赖)

reward_manager.py
├── verl.workers.reward_manager (register, AbstractRewardManager)
├── mathruler.grader (grade_answer, extract_boxed_content)
└── sklearn, nltk (外部依赖)
```

### 4.2 服务依赖

```
ChallengerRewardManager (reward_manager.py)
    │
    │ HTTP 请求 (端口 5000, 5001)
    ▼
start_vllm_server.py (Reward Server)
    │
    │ 调用
    ▼
vLLM Model (Solver 模型)
```

## 五、GPU 分配策略 (当前)

### 5.1 当前硬编码分配

| 阶段 | GPU | 脚本位置 |
|------|-----|----------|
| Challenger 训练 | 4, 5 | challenger.sh:109 |
| Reward Server 1 | 6 | challenger_reward.sh:30 |
| Reward Server 2 | 7 | challenger_reward.sh:32 |
| Query 生成 | 4, 5, 6, 7 | gen_query.sh:53-62 |
| Solver 训练 | 4, 5, 6, 7 | solver.sh:140 |

### 5.2 Reward Server 端口分配 (已修改为动态配置)

```python
# reward_manager.py 中的配置函数（已修改）
def get_reward_server_config():
    """从环境变量获取 Reward Server 配置"""
    ports_str = os.environ.get("SE_REWARD_PORTS", "5000,5001")
    ports = [int(p.strip()) for p in ports_str.split(",") if p.strip()]
    return ports

def fetch(port, filepath, question_reward):
    """向指定端口的 Reward Server 发送请求"""
    response = requests.get(f"http://0.0.0.0:{port}/hello?...")

# generate_results 函数
ports = get_reward_server_config()  # 动态获取端口
n_servers = len(ports)
datas = split_list(data, n_servers)  # 动态分片
```

## 六、存在的问题

### 6.1 导入问题

1. **缺少 `__init__.py`**: `se_code_auto/` 目录没有 `__init__.py`，导致无法作为包导入
2. **混合导入**: `reward.py` 同时导入 `se_code.reward_manager` 和 `se_code_4.reward_manager`
3. **硬编码路径**: 多处使用绝对路径如 `/home/ycy/sdi/Self-evolving-Agent/`

### 6.2 GPU 分配问题

1. 所有 GPU ID 都是硬编码的
2. 没有动态检测可用 GPU 的机制
3. Reward Server 端口数量固定为 2

## 七、修改记录 (已完成)

### 7.1 修复导入问题 ✅

1. ✅ 创建 `se_code_auto/__init__.py`
2. ✅ 修复 `reward.py` 中的导入路径问题

### 7.2 GPU 动态分配 ✅

1. ✅ 创建新启动脚本 `run_with_gpus.sh`
2. ✅ 实现 GPU 检测和等待逻辑
3. ✅ 通过环境变量传递 GPU 配置
4. ✅ 修改 `reward_manager.py` 支持动态端口

### 7.3 文件修改清单

| 操作 | 文件 | 状态 |
|------|------|------|
| 新增 | `se_code_auto/__init__.py` | ✅ |
| 新增 | `se_code_auto/Zero/run_with_gpus.sh` | ✅ |
| 新增 | `se_code_auto/gpu_config.py` | ✅ |
| 新增 | `se_code_auto/README_GPU_LAUNCH.md` | ✅ |
| 修改 | `se_code_auto/Zero/challenger.sh` | ✅ |
| 修改 | `se_code_auto/Zero/solver.sh` | ✅ |
| 修改 | `se_code_auto/Zero/challenger_reward.sh` | ✅ |
| 修改 | `se_code_auto/Zero/gen_query.sh` | ✅ |
| 修改 | `se_code_auto/reward_manager.py` | ✅ |
| 修改 | `se_code_auto/reward.py` | ✅ |

### 7.4 使用方法

```bash
# 新的启动方式
cd /home/ycy/data1/Self-evolving-Agent/se_code_auto/Zero
./run_with_gpus.sh 4   # 使用 4 张 GPU
./run_with_gpus.sh 8   # 使用 8 张 GPU
```

详细使用说明请参考 `README_GPU_LAUNCH.md`
