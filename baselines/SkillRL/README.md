# baselines/SkillRL — DeepMath + 分层技能库 + SFT / RL

自包含副本（**不依赖 `skill_src`**）：`vllm_http_client` 在包内。流程对齐 SkillRL 思路：轨迹 → 技能 JSON → 技能增强 SFT（LLaMA-Factory）→ 技能增强 RL（仓库内 [SkillRL/verl](../../SkillRL)）。

## 依赖

- `httpx`、`pandas`、`pyarrow`（`build-rl-parquet`）
- 可选：`mathruler`（轨迹 `is_correct`）

## CLI（在仓库根目录执行）

```bash
export PYTHONPATH="/path/to/Skill_Evo:${PYTHONPATH}"

python -m baselines.SkillRL gen-traj --data-path ... --start 0 --end 100 --output traj.jsonl
python -m baselines.SkillRL distill --trajectories traj.jsonl --output-skills skills.json
python -m baselines.SkillRL build-sft \
  --deepmath-jsonl ... --skills-json skills.json --trajectories traj.jsonl \
  --start 0 --end 100 --output data/deepmath_skills_sft.json
python -m baselines.SkillRL build-rl-parquet \
  --deepmath-jsonl ... --skills-json skills.json --start 0 --end 1000 \
  --val-ratio 0.05 --output-train train.parquet --output-val val.parquet
python -m baselines.SkillRL inspect --skills-json skills.json --topic your_topic_key
```

## Shell 脚本（[scripts/](scripts/)）

| 脚本 | 作用 |
|------|------|
| [prepare_sft_deepmath.sh](scripts/prepare_sft_deepmath.sh) | 调用 `build-sft`；若设置 `LLAMAFACTORY_HOME` 则拷贝 json 到 `data/deepmath_skills_sft.json` |
| [run_sft_llamafactory.sh](scripts/run_sft_llamafactory.sh) | `cd $LLAMAFACTORY_HOME && llamafactory-cli train`（需已合并 [llamafactory/dataset_info.snippet.json](llamafactory/dataset_info.snippet.json)） |
| [prepare_rl_parquet.sh](scripts/prepare_rl_parquet.sh) | 调用 `build-rl-parquet` 生成 train/val parquet |
| [run_rl_deepmath.sh](scripts/run_rl_deepmath.sh) | `cd $SKILLRL_ROOT` 运行 `verl.trainer.main_ppo`（DAPO reward，无 agent env） |

### SFT 环境变量示例

```bash
export DEEPMATH_JSONL=/data/DeepMath-103K.jsonl
export SKILLS_JSON=./outputs/claude_style_skills.json
export TRAJ=./outputs/traj.jsonl
export START=0 END=5000
export OUT_JSON=./outputs/deepmath_skills_sft.json
export LLAMAFACTORY_HOME=~/LLaMA-Factory
bash baselines/SkillRL/scripts/prepare_sft_deepmath.sh

export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
bash baselines/SkillRL/scripts/run_sft_llamafactory.sh
```

将 `llamafactory/dataset_info.snippet.json` 中的 `deepmath_skills_sft` 合并进 `$LLAMAFACTORY_HOME/data/dataset_info.json`。

### RL 环境变量示例

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH
export DEEPMATH_JSONL=/data/DeepMath-103K.jsonl
export SKILLS_JSON=./outputs/claude_style_skills.json
export START=0 END=10000
bash baselines/SkillRL/scripts/prepare_rl_parquet.sh

export SKILLRL_ROOT=/path/to/Skill_Evo/SkillRL
export TRAIN_FILE=.../deepmath_rl_train.parquet
export TEST_FILE=.../deepmath_rl_val.parquet
export MODEL_PATH=/path/to/sft-merged-or-base
bash baselines/SkillRL/scripts/run_rl_deepmath.sh
```

可按机器调整 `NGPUS_PER_NODE`、`FSDP_SIZE`、`GEN_TP`、`TRAIN_PROMPT_BSZ` 等（见 `run_rl_deepmath.sh` 内变量）。

## 数据契约

- **技能库**：`general_skills` / `task_specific_skills` / `common_mistakes`（与 SkillRL `claude_style_skills.json` 同构）。
- **SFT**：Alpaca 字段 `instruction` / `input` / `output`；`instruction` 由 [prompts/skill_use_math.txt](prompts/skill_use_math.txt) 注入技能块与题目。
- **RL parquet**：`prompt`（chat 列表）、`reward_model.ground_truth`、`data_source=DeepMath-103K`。
