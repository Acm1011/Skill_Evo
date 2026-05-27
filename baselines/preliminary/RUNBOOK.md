# Preliminary Experiments Runbook

这份文档对应 `Skill_Evo/baselines/preliminary` 下的两个实验，目标是方便你在另一台服务器上直接运行。

默认假设：
- 代码仓库根目录是 `.../Skill_Evo`
- `PYTHONPATH` 需要包含 `.../Skill_Evo`
- 你会用本地 rollout server 跑 student / server_teacher
- 外部 API teacher 只在第一个实验里需要

## 实验 1：生成逐题 skill，并评估双 teacher 的效果

对应文件：
- Python: `baselines/preliminary/eval_source_linked_skills.py`
- Bash: `baselines/preliminary/scripts/run_eval_source_linked_skills.sh`

### 实验目的

对 `SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl` 中的题目：
- 按题聚合原始 4 次 rollout
- 用两种 teacher 分别为每题生成一次 source-linked skill
  - `api_teacher`
  - `server_teacher`
- 再把 skill 喂给 student model，对同题 rollout 4 次
- 统计使用 skill 前后的 acc 变化

### 输入

- 轨迹文件：
  - 默认：`Skill_Evo/baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl`
- student / server_teacher 模型：
  - 通过 `--model` 指向本地 checkpoint 或 HF 兼容目录
- 外部 API teacher：
  - `--teacher-api-base-url`
  - `--teacher-api-model`
  - `--teacher-api-key`

### 输出

假设 `--output-dir /path/to/run1`，则输出结构为：

```text
/path/to/run1/
  generated_skills/
    skillrl/api_teacher.jsonl
    skillrl/server_teacher.jsonl
    reasoningbank/api_teacher.jsonl
    reasoningbank/server_teacher.jsonl
    expelmath/api_teacher.jsonl
    expelmath/server_teacher.jsonl
  student_rollout/
    ...
  details.jsonl
  summary.json
```

其中：
- `generated_skills/...` 是后续实验 2 会直接复用的输入
- `details.jsonl` 是逐题 before/after
- `summary.json` 是按 `method + teacher_backend` 聚合的汇总

### 推荐命令

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH

bash Skill_Evo/baselines/preliminary/scripts/run_eval_source_linked_skills.sh \
  --model /path/to/student_or_base_model \
  --output-dir /path/to/run1 \
  --teacher-api-base-url http://your-api-host:8000/v1 \
  --teacher-api-model your-teacher-model \
  --teacher-api-key your-key \
  --sample-size 5000 \
  --gpu-ids 0,1,2,3 \
  --n-gpus 4
```

### 常用可选参数

- `--trajectories /path/to/trajectories.jsonl`
- `--method skillrl|reasoningbank|expelmath|all`
- `--sample-size 5000`
- `--student-rollout-n 4`
- `--rollout-host 127.0.0.1`
- `--rollout-base-port 8760`
- `--eval-max-workers 0`

说明：
- `--n-gpus` 负责启动多少个 rollout server
- `--eval-max-workers` 负责题级并发数；默认 `0` 时会自动取 rollout server 数量

### 运行前检查

- `trajectories_from_merged_v1_v2.jsonl` 确实存在
- student 模型目录可被 rollout server 正常加载
- 外部 API teacher 是 OpenAI-compatible `/chat/completions`
- GPU 数与 `--gpu-ids`、`--n-gpus` 一致

## 实验 2：复用同一批 skill，追踪不同 checkpoint 上的 acc 漂移

对应文件：
- Python: `baselines/preliminary/eval_skill_drift_across_checkpoints.py`
- Bash: `baselines/preliminary/scripts/run_eval_skill_drift_across_checkpoints.sh`

### 实验目的

复用实验 1 已经生成好的逐题 skill，不再重新生成 teacher skill。

对训练过程中的多个 student checkpoint：
- 对每个 checkpoint 先跑无 skill baseline 4 次
- 再对同题使用同一批 skill 跑 4 次
- 观察同一个 skill 在不同训练阶段的 acc / delta 是否变化

### 输入

- 实验 1 输出目录：
  - 通过 `--skills-run-dir /path/to/run1`
  - 其中必须存在 `generated_skills/<method>/<teacher_backend>.jsonl`
- checkpoint 根目录：
  - 通过 `--checkpoint-root /path/to/checkpoints_root`
  - 脚本会自动递归扫描 checkpoint
- 轨迹文件：
  - 默认仍是 `SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl`

### checkpoint 自动发现规则

脚本会递归扫描 `--checkpoint-root` 下看起来像模型目录的子目录。

排序优先按目录名中的数字规则：
- `checkpoint-100`
- `checkpoint-200`
- `global_step_500`
- `step_1000`
- `epoch_3`

如果目录名里提不出数值，就回退到字典序。

### 输出

假设 `--output-dir /path/to/run2`，则输出结构为：

```text
/path/to/run2/
  per_checkpoint/
    checkpoint-100/
      details.jsonl
      attempts.jsonl
      summary.json
    checkpoint-200/
      details.jsonl
      attempts.jsonl
      summary.json
  cross_checkpoint_details.jsonl
  cross_checkpoint_attempts.jsonl
  cross_checkpoint_summary.json
```

其中：
- `per_checkpoint/.../summary.json` 是单个 checkpoint 下各 skill 集的汇总
- `cross_checkpoint_summary.json` 是跨 checkpoint 的趋势表
- `cross_checkpoint_details.jsonl` 适合后续画曲线

### 推荐命令

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH

bash Skill_Evo/baselines/preliminary/scripts/run_eval_skill_drift_across_checkpoints.sh \
  --skills-run-dir /path/to/run1 \
  --checkpoint-root /path/to/checkpoints_root \
  --output-dir /path/to/run2 \
  --sample-size 5000 \
  --checkpoint-limit 0 \
  --gpu-ids 0,1,2,3 \
  --n-gpus 4
```

### 调试时建议先小跑

```bash
bash Skill_Evo/baselines/preliminary/scripts/run_eval_skill_drift_across_checkpoints.sh \
  --skills-run-dir /path/to/run1 \
  --checkpoint-root /path/to/checkpoints_root \
  --output-dir /path/to/run2_debug \
  --sample-size 20 \
  --checkpoint-limit 2 \
  --gpu-ids 0,1 \
  --n-gpus 2
```

### 常用可选参数

- `--trajectories /path/to/trajectories.jsonl`
- `--methods skillrl,reasoningbank,expelmath`
- `--teacher-backends api_teacher,server_teacher`
- `--student-rollout-n 4`
- `--checkpoint-limit 2`
- `--rollout-host 127.0.0.1`
- `--rollout-base-port 8760`

## 两个实验的关系

推荐顺序：

1. 先跑实验 1，得到一份固定的 `generated_skills`
2. 再把实验 1 的 `output-dir` 传给实验 2 的 `--skills-run-dir`
3. 用实验 2 对多个训练 checkpoint 做追踪

简化理解：
- 实验 1 研究：skill 本身是否有用
- 实验 2 研究：同一个 skill 在训练前后是否变得更有用或更没用

## 常见替换项

你在另一台服务器上通常只需要改这些：

- `/path/to/Skill_Evo`
- `/path/to/student_or_base_model`
- `/path/to/run1`
- `/path/to/run2`
- `/path/to/checkpoints_root`
- `http://your-api-host:8000/v1`
- `your-teacher-model`
- `your-key`
- `--gpu-ids`
- `--n-gpus`

## 最后检查

实验 1 跑完后，先确认：

- `generated_skills/skillrl/api_teacher.jsonl` 存在
- `generated_skills/skillrl/server_teacher.jsonl` 存在
- `summary.json` 非空

实验 2 跑完后，先确认：

- `per_checkpoint/<checkpoint_name>/summary.json` 存在
- `cross_checkpoint_summary.json` 非空
- 同一个 `method + teacher_backend` 在 `cross_checkpoint_summary.json` 中有多个 checkpoint 记录

## 常见错误

如果日志里反复出现：

```text
Server http://127.0.0.1:8760: {"detail":"Not Found"}
```

通常表示客户端在访问 OpenAI-compatible `/v1/completions`，但 `start_rollout_servers.sh` 启动的是 `solver_offline_rollout_server`，它提供的是 `/rollout`。当前 `preliminary` 两个实验已经统一使用 `/rollout` 协议；如果仍看到这个错误，先确认运行的是最新的：

```bash
python -m baselines.preliminary.eval_source_linked_skills ...
python -m baselines.preliminary.eval_skill_drift_across_checkpoints ...
```
