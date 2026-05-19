# baselines/RetroAgent — 数学单回合适配版

这个目录是 `RetroAgent/rl_trained_self_reflection` 的 **math overlay baseline**。

目标不是复刻 WebShop / ALFWorld 那种长交互环境，而是把 RetroAgent 的两部分核心机制迁移到数学题：

- hindsight self-reflection
- reflection memory retrieval + utility update

当前版本固定为：

- 只支持 `rl_trained_self_reflection`
- 只支持 **单题最终答案** 数学任务
- play phase 退化为 **单次答案提交**
- `env.max_steps=1`
- 判分默认走上游 `prime_math`

## 目录结构

- `agent_system/`：本 baseline 的 overlay，实现 math 专用环境；`memory/`、`multi_turn_rollout/`、`reward_manager/` 基于上游最小副本
- `config/ppo_trainer_math.yaml`：Hydra 配置
- `build_math_parquet.py`：DeepMath 风格 jsonl 转 RetroAgent parquet
- `run_ppo_math.py`：训练入口
- `scripts/`：数据准备与训练脚本
- `tests/`：基础单测

## 运行方式

先准备 parquet：

```bash
export DEEPMATH_JSONL=/path/to/DeepMath-103K.jsonl
export END=10000
export OUT_TRAIN=/path/to/deepmath_rl_train.parquet
export OUT_VAL=/path/to/deepmath_rl_val.parquet
export VAL_RATIO=0.05
bash baselines/RetroAgent/scripts/prepare_math_parquet.sh
```

再启动训练：

```bash
export TRAIN_FILE=/path/to/deepmath_rl_train.parquet
export VAL_FILE=/path/to/deepmath_rl_val.parquet
export MODEL_PATH=/path/to/model
bash baselines/RetroAgent/scripts/run_rl_retro_math.sh
```

## 数据格式

输出 parquet 固定包含：

- `data_source`
- `prompt`
- `ability = "math"`
- `reward_model.ground_truth`
- `extra_info`
- `env_kwargs`

其中真实题目不从 `prompt` 驱动，而是从 `env_kwargs.question` 注入环境。

`env_kwargs` 结构固定为：

```json
{
  "question": "...",
  "ground_truth": "...",
  "data_source": "DeepMath-103K",
  "topic": "algebra",
  "index": 123
}
```

## 环境协议

环境名固定为 `MathSingleTurn`。

play phase 输出协议：

- 推理放在 `<think>...</think>`
- 最终答案放在 `<answer>...</answer>`

环境在收到首个 `<answer>` 后立即结束本题并判分。

## 反思协议

反思阶段要求模型输出 JSON，固定字段：

- `subtasks`
- `task_success`
- `action_lesson`
- `reasoning_lesson`

`subtasks` 使用固定五个槽位：

- `understand_problem`
- `choose_strategy`
- `derive_solution`
- `check_constraints`
- `finalize_answer`

## 说明

这不是原论文交互环境的等价复现，而是 **RetroAgent 思想在数学单回合任务上的 v1 迁移**。

如果后续要做 search / tool-augmented 数学，应作为第二阶段扩展，不放进当前 baseline。
