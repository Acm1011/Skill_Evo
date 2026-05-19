# baselines/ExpeLMath

数学版 ExpeL baseline，面向 `DeepMath` / `SkillRL` 轨迹，核心聚焦：

- 从成功/失败轨迹抽取 ExpeL-style reusable insights
- 将 insight 同时保存为 `raw_rule` 与结构化 `memory_items`
- 用 embedding + topic/rule-type 重排检索 memory
- 在数学推理时把 memory 注入 prompt

## CLI

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH

python -m baselines.ExpeLMath gen-traj --data-path ... --end 100 --output traj.jsonl
python -m baselines.ExpeLMath build-memory --trajectories traj.jsonl --output memory_bank.jsonl
python -m baselines.ExpeLMath build-embeddings --memory-bank memory_bank.jsonl --output memory_embeddings.jsonl
python -m baselines.ExpeLMath retrieve --memory-bank memory_bank.jsonl --embeddings memory_embeddings.jsonl --question "..."
python -m baselines.ExpeLMath eval --deepmath-jsonl ... --memory-bank memory_bank.jsonl --embeddings memory_embeddings.jsonl --end 100 --output eval.jsonl
python -m baselines.ExpeLMath evolve-memory --memory-bank memory_bank.jsonl --trajectories new_traj.jsonl --output-memory-bank memory_bank_v2.jsonl
```

## Data Contract

- 输入轨迹兼容 `SkillRL`：
  - `problem`
  - `topic`
  - `topic_key`
  - `student_response`
  - `is_correct`
  - `ground_truth`
- memory 记录字段：
  - `memory_id`
  - `source_idx`
  - `query`
  - `topic`
  - `topic_key`
  - `status`
  - `memory_type`
  - `trajectory`
  - `raw_rule`
  - `memory_items`
  - `embedding_text`
  - `provenance`
  - `duplicate_count`

## Memory Types

- `compare_rule`: 同题成功/失败对比得到的规则
- `success_rule`: 跨题成功轨迹归纳出的正向规则
- `failure_rule`: 多条失败轨迹归纳出的避免性规则
