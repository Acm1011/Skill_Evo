## SkillOpt DeepMath Baseline 中文运行指南

本文档说明如何运行当前仓库中已经适配好的 `SkillOpt` DeepMath/数学任务 baseline。该 baseline 保留 `SkillOpt` 原始的在线技能优化范式：从数学题数据中采样题目，使用当前 skill 调用目标模型重新解题，再根据答案正确性进行反思、聚合、选择和更新，最终得到优化后的 `best_skill.md`。

### 1. 适配内容概览

关键文件如下：

- **训练入口**：`scripts/train.py`
- **DeepMath 配置**：`configs/deepmath/default.yaml`
- **DeepMath 环境适配器**：`skillopt/envs/deepmath/adapter.py`
- **DeepMath 数据读取**：`skillopt/envs/deepmath/dataloader.py`
- **DeepMath rollout**：`skillopt/envs/deepmath/rollout.py`
- **DeepMath 评测**：`skillopt/envs/deepmath/evaluator.py`
- **初始 skill**：`skillopt/envs/deepmath/skills/initial.md`
- **训练后数据生成脚本**：`scripts/prepare_prompt_data.py`

### 2. 默认输入数据

默认配置文件 `configs/deepmath/default.yaml` 中已经指定：

```yaml
env:
  name: deepmath
  data_path: ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl
  split_mode: ratio
  split_ratio: 8:1:1
```

从 `baselines/SkillOpt` 目录运行时，默认会读取：

```text
../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl
```

`SkillOpt` 会把该 JSONL 当作数学题数据池，而不是直接把已有轨迹总结成 skill。它主要读取 `problem` / `question` / `raw_question`、`ground_truth` / `gt` / `reward_model.ground_truth`、`topic` / `topic_key` / `extra_info.topic`。如果原始 JSONL 中有 `student_response` 和 `is_correct`，会保存在样本元信息里，但训练主流程仍会使用当前 skill 重新调用目标模型解题，然后根据新 response 和 ground truth 做评测。

### 3. 环境准备

进入 `SkillOpt` 根目录：

```bash
cd /apdcephfs/cq/apdcephfs_cq10/share_1567347/share_info/cloudytang/Skill_Evo/baselines/SkillOpt
```

安装基础依赖：

```bash
python3 -m pip install -r requirements.txt
```

如果需要输出 Parquet 或更强的数学答案判分能力，建议额外安装：

```bash
python3 -m pip install pandas pyarrow mathruler
```

其中 `pandas` / `pyarrow` 用于输出 Parquet，`mathruler` 用于更稳健地比较数学答案；如果未安装，会降级为 `<answer>` / `\boxed{}` 抽取后的字符串匹配。

### 4. 快速检查数据路径

训练前建议确认默认数据文件存在：

```bash
test -f ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl && echo "data ok"
```

如果数据不在默认位置，可以在训练命令中使用 `--data_path` 或 `--cfg-options env.data_path=/abs/path/file.jsonl` 覆盖。

### 5. 运行 smoke test

建议先用小样本快速验证流程是否打通：

```bash
python3 scripts/train.py \
  --config configs/deepmath/default.yaml \
  --backend qwen_chat \
  --optimizer_model /apdcephfs_cq10/share_1567347/share_info/cloudytang/model/Qwen3-30B-A3B-Instruct-2507 \
  --target_model /apdcephfs_cq10/share_1567347/share_info/cloudytang/model/Qwen3-30B-A3B-Instruct-2507 \
  --limit 64 \
  --num_epochs 1 \
  --batch_size 8 \
  --workers 8 \
  --out_root outputs/deepmath_skillopt_smoke
```

常用参数含义：

- **`--optimizer_model`**：用于反思、分析和更新 skill 的模型。
- **`--target_model`**：用于在数学题上重新 rollout 的模型。
- **`--limit`**：限制读取样本数，调试时建议设置。
- **`--workers`**：target model rollout 并发数。
- **`--out_root`**：本次运行输出目录。

### 6. 正式运行训练

确认 smoke test 正常后，可以扩大样本和并发：

```bash
python3 scripts/train.py \
  --config configs/deepmath/default.yaml \
  --backend qwen_chat \
  --optimizer_model /apdcephfs_cq10/share_1567347/share_info/cloudytang/model/Qwen3-30B-A3B-Instruct-2507 \
  --target_model /apdcephfs_cq10/share_1567347/share_info/cloudytang/model/Qwen3-30B-A3B-Instruct-2507 \
  --num_epochs 3 \
  --batch_size 32 \
  --workers 32 \
  --out_root outputs/deepmath_skillopt_run
```

也可以使用结构化覆盖参数：

```bash
python3 scripts/train.py \
  --config configs/deepmath/default.yaml \
  --backend qwen_chat \
  --optimizer_model /path/to/optimizer_model \
  --target_model /path/to/target_model \
  --cfg-options \
    train.num_epochs=2 \
    train.batch_size=16 \
    env.limit=256 \
    env.workers=16 \
    env.data_path=/path/to/deepmath_or_trajectory.jsonl \
    env.out_root=outputs/deepmath_skillopt_custom
```

### 7. 主要输出文件

训练完成后，重点查看 `--out_root` 目录，例如 `outputs/deepmath_skillopt_run`：

- **`best_skill.md`**：最终选择出的最优 skill，后续生成 temp/greedy 数据通常使用这个文件。
- **`skills/skill_vXXXX.md`**：训练过程中每一步保存的 skill 快照。
- **`epoch_*/step_*/results.jsonl`**：每个 batch 的 rollout 和评测结果。
- **`epoch_*/step_*/predictions/`**：每道题的 prompt、response、评测信息等明细。
- **`summary.json`**：训练过程汇总指标。

### 8. 生成带 SkillOpt memory 的 temp / greedy 数据

训练结束后，可以用 `scripts/prepare_prompt_data.py` 把 `best_skill.md` 注入到新的 DeepMath 输入数据中，生成后续评测或训练可用的 JSONL / Parquet。

生成 temp 数据示例：

```bash
python3 scripts/prepare_prompt_data.py \
  --input-jsonl /path/to/temp_input.jsonl \
  --skill-file outputs/deepmath_skillopt_run/best_skill.md \
  --output-jsonl outputs/deepmath_skillopt_temp.jsonl \
  --output-parquet outputs/deepmath_skillopt_temp.parquet \
  --data-source SkillOptMath \
  --keep-raw-prompt
```

生成 greedy 数据示例：

```bash
python3 scripts/prepare_prompt_data.py \
  --input-jsonl /path/to/greedy_input.jsonl \
  --skill-file outputs/deepmath_skillopt_run/best_skill.md \
  --output-jsonl outputs/deepmath_skillopt_greedy.jsonl \
  --output-parquet outputs/deepmath_skillopt_greedy.parquet \
  --data-source SkillOptMath \
  --keep-raw-prompt
```

如果只需要 JSONL，可以去掉 `--output-parquet`。如果只想处理一部分数据，可以使用 `--start 0 --end 1000`。

### 9. 流程示意

```text
SkillRL trajectory / DeepMath JSONL
        ↓
DeepMathDataLoader 读取题目、答案和 topic
        ↓
按 train / val / test 切分
        ↓
当前 skill + 数学题 → target model 重新 rollout
        ↓
抽取 <answer> / \boxed{} 并和 ground_truth 判分
        ↓
SkillOpt 反思、聚合、选择、更新 skill
        ↓
best_skill.md
        ↓
scripts/prepare_prompt_data.py
        ↓
带 SkillOpt memory 的 temp / greedy JSONL 或 Parquet
```

### 10. 常见问题

- **找不到默认数据文件**：确认当前目录是 `baselines/SkillOpt`，并检查 `../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl` 是否存在；也可以用 `--data_path /abs/path/file.jsonl` 覆盖。
- **运行很慢**：先用 `--limit 64 --num_epochs 1 --batch_size 8 --workers 8` 做 smoke test，再逐步增加规模。
- **答案判分不准**：优先安装 `mathruler`，并检查数据里的 `ground_truth` 字段是否规范。
- **没有生成 Parquet**：安装 `pandas` 和 `pyarrow`，或者只使用 `--output-jsonl`。
- **找不到 `best_skill.md`**：建议每次运行显式传 `--out_root`，训练结束后在该目录下查看。
