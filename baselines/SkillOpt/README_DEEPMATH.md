## SkillOpt DeepMath 运行指南

本文档说明当前 `SkillOpt` baseline 如何在 DeepMath/数学任务上运行。该适配保留了 `SkillOpt` 原始的在线技能优化流程：读取数学题数据，使用当前 skill 重新 rollout，基于答案正确性做反思和 patch/rewrite，再输出优化后的 skill，并可将最终 skill 注入 temp/greedy 数据。

### 1. 目录与关键文件

- **训练入口**：`scripts/train.py`
- **DeepMath 配置**：`configs/deepmath/default.yaml`
- **DeepMath 环境适配器**：`skillopt/envs/deepmath/adapter.py`
- **DeepMath 数据读取**：`skillopt/envs/deepmath/dataloader.py`
- **DeepMath rollout 与评测**：`skillopt/envs/deepmath/rollout.py`、`skillopt/envs/deepmath/evaluator.py`
- **初始 skill**：`skillopt/envs/deepmath/skills/initial.md`
- **训练后 prompt 数据生成**：`scripts/prepare_prompt_data.py`

### 2. 默认数据来源

默认配置读取：

```text
../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl
```

对应配置位于 `configs/deepmath/default.yaml`：

```yaml
env:
  name: deepmath
  data_path: ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl
  split_mode: ratio
  split_ratio: 8:1:1
```

`SkillOpt` 会把该 JSONL 当作 DeepMath 题目数据池读取，主要使用以下字段：

- `problem` / `question` / `raw_question`
- `ground_truth` / `gt` / `reward_model.ground_truth`
- `topic` / `topic_key` / `extra_info.topic`

如果原始轨迹中存在 `student_response`、`is_correct`，会保留到样本元信息里，但 `SkillOpt` 训练时会用当前 skill 重新调用目标模型解题，而不是直接复用旧 response 做最终评测。

### 3. 环境准备

在 `SkillOpt` 根目录执行：

```bash
cd /apdcephfs/cq/apdcephfs_cq10/share_1567347/share_info/cloudytang/Skill_Evo/baselines/SkillOpt
pip install -r requirements.txt
```

如果需要更准确的数学答案判分，建议安装 `mathruler`。未安装时会降级为 `<answer>` / `\boxed{}` 抽取后的字符串匹配。

### 4. 运行 DeepMath 训练

推荐显式指定输出目录，便于后续找 `best_skill.md`：

```bash
python3 scripts/train.py \
  --config configs/deepmath/default.yaml \
  --backend qwen_chat \
  --optimizer_model /apdcephfs_cq10/share_1567347/share_info/cloudytang/model/Qwen3-30B-A3B-Instruct-2507 \
  --target_model /apdcephfs_cq10/share_1567347/share_info/cloudytang/model/Qwen3-30B-A3B-Instruct-2507 \
  --out_root outputs/deepmath_skillopt_run
```

如果要快速 smoke test，可以限制数据量和训练轮数：

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

也可以使用 `--cfg-options` 覆盖结构化配置，例如：

```bash
python3 scripts/train.py \
  --config configs/deepmath/default.yaml \
  --cfg-options train.num_epochs=2 train.batch_size=16 env.limit=256 env.workers=16 env.data_path=/path/to/deepmath.jsonl
```

### 5. 训练输出

训练完成后，主要输出在 `--out_root` 指定目录下：

- `best_skill.md`：最终选择出的最优 skill，后续生成 temp/greedy prompt 数据时通常使用这个文件。
- `skills/skill_vXXXX.md`：每一步更新后的 skill 快照。
- `epoch_*/step_*/results.jsonl`：每个 batch 的 rollout 结果。
- `epoch_*/step_*/predictions/`：每个样本的 prompt、response、评测信息。
- `summary.json` 或训练日志：训练过程中的 hard/soft 指标和选择结果。

### 6. 生成 temp / greedy prompt 数据

训练完后，用 `scripts/prepare_prompt_data.py` 将 `best_skill.md` 注入 DeepMath 输入数据。该脚本支持 JSONL 输出，也可选输出 Parquet。

生成 temp 数据示例：

```bash
python3 scripts/prepare_prompt_data.py \
  --input-jsonl /path/to/temp_input.jsonl \
  --skill-file outputs/deepmath_skillopt_run/best_skill.md \
  --output-jsonl outputs/deepmath_skillopt_temp.jsonl \
  --output-parquet outputs/deepmath_skillopt_temp.parquet \
  --data-source SkillOptMath
```

生成 greedy 数据示例：

```bash
python3 scripts/prepare_prompt_data.py \
  --input-jsonl /path/to/greedy_input.jsonl \
  --skill-file outputs/deepmath_skillopt_run/best_skill.md \
  --output-jsonl outputs/deepmath_skillopt_greedy.jsonl \
  --output-parquet outputs/deepmath_skillopt_greedy.parquet \
  --data-source SkillOptMath
```

如果只需要 JSONL，可以去掉 `--output-parquet`。如果环境没有 `pandas` / `pyarrow`，Parquet 会跳过或失败，JSONL 不受影响。

### 7. 常用参数

- `--data_path`：覆盖 DeepMath/trajectory JSONL 路径。
- `--split_mode`：`ratio` 表示从单个 JSONL 按比例切分；`split_dir` 表示使用已有 train/val/test split。
- `--split_ratio`：默认 `8:1:1`。
- `--limit`：限制读取样本数，调试时很有用。
- `--num_epochs`：训练轮数。
- `--batch_size`：每个训练 batch 的样本数。
- `--workers`：target model rollout 并发数。
- `--analyst_workers`：反思/分析并发数。
- `--failure_only`：是否只分析失败样本。
- `--out_root`：输出目录。

### 8. 完整流程示意

```text
SkillRL trajectory / DeepMath JSONL
        ↓
DeepMathDataLoader 按 train/val/test 切分
        ↓
当前 skill + 题目 → target model rollout
        ↓
抽取 <answer> / \boxed{} 并和 ground_truth 判分
        ↓
SkillOpt reflect / aggregate / select / update
        ↓
best_skill.md
        ↓
scripts/prepare_prompt_data.py
        ↓
带 SkillOpt memory 的 temp / greedy JSONL 或 Parquet
```

### 9. 常见问题

- **找不到数据文件**：确认从 `SkillOpt` 根目录运行时，`../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl` 是否存在；也可以用 `--data_path /abs/path/file.jsonl` 覆盖。
- **训练很慢**：先用 `--limit 64 --num_epochs 1 --batch_size 8 --workers 8` 做 smoke test。
- **答案判分不准**：安装 `mathruler`，或检查数据里的 `ground_truth` 是否是标准答案字段。
- **输出目录找不到 `best_skill.md`**：建议运行时显式传 `--out_root outputs/deepmath_skillopt_run`。
