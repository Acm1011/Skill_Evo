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
