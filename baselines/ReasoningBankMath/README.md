# baselines/ReasoningBankMath

数学版 ReasoningBank 基线，聚焦 `DeepMath` / `SkillRL` 风格轨迹上的 memory 库建立、embedding 检索与增量进化。

## CLI

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH

python -m baselines.ReasoningBankMath gen-traj --data-path ... --end 100 --output traj.jsonl
python -m baselines.ReasoningBankMath build-memory \
  --trajectories traj.jsonl --output memory_bank.jsonl \
  --teacher-base-url http://127.0.0.1:8000/v1 --teacher-model your-model
python -m baselines.ReasoningBankMath build-embeddings \
  --memory-bank memory_bank.jsonl --output memory_embeddings.jsonl --backend hash
python -m baselines.ReasoningBankMath retrieve \
  --memory-bank memory_bank.jsonl --embeddings memory_embeddings.jsonl \
  --question "..." --topic "Math->Algebra" --top-k 3
python -m baselines.ReasoningBankMath evolve-memory \
  --memory-bank memory_bank.jsonl --trajectories new_traj.jsonl \
  --output-memory-bank memory_bank_v2.jsonl \
  --existing-embeddings memory_embeddings.jsonl \
  --output-embeddings memory_embeddings_v2.jsonl \
  --teacher-base-url http://127.0.0.1:8000/v1 --teacher-model your-model
```

## 什么时候需要启动 server

- `gen-traj`：需要 rollout server，因为它会调用学生模型生成轨迹。
- `build-memory` / `evolve-memory`：
  - 用 `--teacher-backend chat` 时，不需要 rollout server，但需要一个 OpenAI-compatible `/chat/completions` 服务。
  - 用 `--teacher-backend rollout` 时，需要 rollout server。
- `build-embeddings`：
  - 用 `--backend hash` 时，不需要任何服务。
  - 用 `--backend openai` 时，需要一个 OpenAI-compatible `/embeddings` 服务。
- `retrieve`：只依赖已经生成好的 embedding 文件；若现查现嵌入且 `--backend openai`，则需要 embedding 服务。

## 直接可执行脚本

脚本目录：[scripts/](scripts)

- `run_gen_traj.sh`：自动启动 `skill_src/Zero/start_rollout_servers.sh`，然后生成轨迹。
- `run_build_memory_with_rollout.sh`：自动启动 rollout server，然后用 rollout backend 建 memory 库。
- `run_build_memory_pipeline.sh`：自动启动 rollout server，然后从 `trajectories jsonl` 一次性生成 `memory_bank + memory_embeddings`。
- `run_evolve_memory_with_rollout.sh`：自动启动 rollout server，然后增量进化 memory 库。
- `run_compact_memory.sh`：不依赖新轨迹，直接对已有 `memory_bank` 做去重合并，并可同步刷新 embeddings。
- `run_refine_memory_with_rollout.sh`：先聚类相近 memory，再调用 LLM 对每个 cluster 重写成更干净的 evolved memory。
- `build_embeddings.sh`：直接包装 `python -m baselines.ReasoningBankMath build-embeddings`
- `retrieve.sh`：直接包装 `python -m baselines.ReasoningBankMath retrieve`

示例：

```bash
bash baselines/ReasoningBankMath/scripts/run_build_memory_with_rollout.sh \
  --trajectories baselines/SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output baselines/ReasoningBankMath/outputs/memory_bank_v1_v2.jsonl

bash baselines/ReasoningBankMath/scripts/run_build_memory_pipeline.sh \
  --model ../models/Qwen3-4B-Instruct-2507

bash baselines/ReasoningBankMath/scripts/run_compact_memory.sh \
  --memory-bank baselines/ReasoningBankMath/outputs/memory_bank_v1_v2.jsonl \
  --output-memory-bank baselines/ReasoningBankMath/outputs/memory_bank_v1_v2_compact.jsonl \
  --existing-embeddings baselines/ReasoningBankMath/outputs/memory_embeddings_v1_v2.jsonl \
  --output-embeddings baselines/ReasoningBankMath/outputs/memory_embeddings_v1_v2_compact.jsonl

bash baselines/ReasoningBankMath/scripts/run_refine_memory_with_rollout.sh \
  --memory-bank baselines/ReasoningBankMath/outputs/memory_bank_v1_v2.jsonl \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --existing-embeddings baselines/ReasoningBankMath/outputs/memory_embeddings_v1_v2.jsonl \
  --output-memory-bank baselines/ReasoningBankMath/outputs/memory_bank_v1_v2_refined.jsonl \
  --output-embeddings baselines/ReasoningBankMath/outputs/memory_embeddings_v1_v2_refined.jsonl
```

## Data Contract

- 输入轨迹兼容 `SkillRL`：`problem`、`topic`、`topic_key`、`difficulty`、`student_response`、`is_correct`、`ground_truth`
- memory 记录字段：
  - `memory_id`
  - `source_idx`
  - `query`
  - `topic`
  - `topic_key`
  - `status`
  - `trajectory`
  - `memory_items`
  - `created_at`
  - `updated_at`
  - `embedding_text`
  - `provenance`
  - `duplicate_count`

## Backends

- Teacher：OpenAI-compatible `/chat/completions`
- Embedding：
  - `hash`：默认，无外部依赖，便于测试
  - `openai`：OpenAI-compatible `/embeddings`
