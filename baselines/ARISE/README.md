# baselines/ARISE

这个目录提供两类能力：

- `DeepMath-103K.jsonl -> ARISE/verl` 原生训练 parquet
- `ARISE` 的 skill library 适配到当前仓库 `baselines` 体系下的数据准备层

如果你要从零复现 ARISE，先用这里的原始 parquet 生成脚本，再进入 [`ARISE`](../../ARISE) 目录用它自带的 `verl` 启动。

当前覆盖两件事：

- `DeepMath-103K.jsonl` + ARISE library checkpoint -> RL 训练用 `jsonl/parquet`
- `temp_data.jsonl` / `greedy_data.jsonl` + ARISE library checkpoint -> 检索 top-k skills 后的 `*_skill.jsonl/parquet`

## 从零训练：原始 DeepMath -> ARISE parquet

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH
export DEEPMATH_JSONL=/home/ycy/sdi/data/DeepMath-103K.jsonl
export OUT_PARQUET=baselines/ARISE/outputs/deepmath_raw_arise.parquet

bash baselines/ARISE/scripts/prepare_raw_deepmath_parquet.sh
```

这一步不会注入任何外部 skill。
它只做最小格式转换，使数据满足 `ARISE/verl` 默认输入契约：

- `prompt`
- `reward_model.ground_truth`
- `data_source=math_dapo`
- `extra_info`

然后进入 `ARISE/` 目录训练：

```bash
cd ARISE
export ARISE_MODEL_PATH=/path/to/model
export ARISE_TRAIN_FILE=/path/to/Skill_Evo/baselines/ARISE/outputs/deepmath_raw_arise.parquet
export ARISE_CKPTS_DIR=/path/to/checkpoints

bash recipe/arise/run_arise_qwen3_4b.sh data.val_files=[]
```

## 输入 skill library

使用 ARISE 训练过程中导出的 library checkpoint json，格式应包含：

- `cache`
- 可选 `reservoir`
- 每条记录内的 `document.skill_name / problem_type / key_insight / method / check`

默认只从 `cache` 检索；加 `--include-reservoir` 可把 `reservoir` 一起作为候选池。

## 训练数据

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH
export ARISE_LIBRARY_JSON=/path/to/library_step_0500.json

bash baselines/ARISE/scripts/prepare_rl_parquet.sh
```

默认输入是 `/home/ycy/sdi/data/DeepMath-103K.jsonl`，输出到：

- `baselines/ARISE/outputs/deepmath_arise_rl.jsonl`
- `baselines/ARISE/outputs/deepmath_arise_rl.parquet`

也可以直接：

```bash
python -m baselines.ARISE prepare-rl-data \
  --deepmath-jsonl /home/ycy/sdi/data/DeepMath-103K.jsonl \
  --library-json /path/to/library_step_0500.json \
  --top-k 3 \
  --output-jsonl baselines/ARISE/outputs/deepmath_arise_rl.jsonl \
  --output-parquet baselines/ARISE/outputs/deepmath_arise_rl.parquet
```

## temp / greedy 适配

```bash
bash baselines/ARISE/scripts/run_prepare_prompt_data.sh \
  --library-json /path/to/library_step_0500.json \
  --top-k 3
```

默认读取：

- `/home/ycy/sdi/data/temp_data.jsonl`
- `/home/ycy/sdi/data/greedy_data.jsonl`

输出到：

- `baselines/ARISE/outputs/prepared/temp_data_skill.jsonl`
- `baselines/ARISE/outputs/prepared/temp_data_skill.parquet`
- `baselines/ARISE/outputs/prepared/greedy_data_skill.jsonl`
- `baselines/ARISE/outputs/prepared/greedy_data_skill.parquet`

## 说明

- 检索仍复用现有 retriever 服务接口 `/rank`，默认 `http://127.0.0.1:8766`
- prompt 模板与其他数学 baseline 保持一致：`SKILL: {skill}\nQuestion: {question}`
- 这次迁移不接入 `ARISE/verl` 分叉代码；如果后续要把 ARISE 的层级训练逻辑真正并进根目录 `verl`，需要单独做 trainer 级集成
