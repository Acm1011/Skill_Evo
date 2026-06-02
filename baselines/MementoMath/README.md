# baselines/MementoMath

数学版 Memento baseline，面向 `DeepMath` / `SkillRL` 轨迹，重点是把数学题轨迹沉淀成一个可检索、可增量更新的 memory 库。

核心产物：

- `memory_bank.jsonl`：结构化 memory 主库
- `memory_embeddings.jsonl`：memory 检索向量
- `memory.jsonl` 风格 case pool：`case / plan / case_label`
- `dummy_memo.jsonl` 风格导出：`question / plan / reward`
- `training_data.jsonl`：给 Memento-style retriever 训练用的 pair 数据
- `retriever_ckpts/`：原始 Memento parametric retriever 的 checkpoint 输出目录

## CLI

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH

python -m baselines.MementoMath gen-traj \
  --data-path /path/to/DeepMath-103K.jsonl \
  --start 0 --end 100 \
  --output baselines/MementoMath/outputs/traj.jsonl \
  --server-urls http://127.0.0.1:8760 http://127.0.0.1:8761

python -m baselines.MementoMath build-memory \
  --teacher-backend rollout \
  --trajectories baselines/MementoMath/outputs/traj.jsonl \
  --output baselines/MementoMath/outputs/memory_bank.jsonl \
  --case-pool-output baselines/MementoMath/outputs/memory.jsonl \
  --dummy-memory-output baselines/MementoMath/outputs/dummy_memo.jsonl \
  --rollout-host 127.0.0.1 --rollout-base-port 8760 --rollout-n-servers 2

python -m baselines.MementoMath build-embeddings \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --output baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --backend hash

python -m baselines.MementoMath build-training-data \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --embeddings baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --output baselines/MementoMath/outputs/training_data.jsonl

python -m baselines.MementoMath retrieve \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --embeddings baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --question "Solve x^2-5x+6=0" \
  --topic "Math->Algebra" \
  --status success \
  --top-k 4

python -m baselines.MementoMath train-retriever \
  --train baselines/MementoMath/outputs/training_data.jsonl \
  --output-dir baselines/MementoMath/outputs/retriever_ckpts \
  --use-plan --save-best

python -m baselines.MementoMath evolve-memory \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --trajectories baselines/MementoMath/outputs/new_traj.jsonl \
  --output-memory-bank baselines/MementoMath/outputs/memory_bank_v2.jsonl \
  --output-case-pool baselines/MementoMath/outputs/memory_v2.jsonl \
  --output-dummy-memory baselines/MementoMath/outputs/dummy_memo_v2.jsonl \
  --existing-embeddings baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --output-embeddings baselines/MementoMath/outputs/memory_embeddings_v2.jsonl
```

## 直接可执行脚本

脚本目录：[scripts/](scripts)

- `run_gen_traj.sh`：自动启动 `skill_src/Zero/start_rollout_servers.sh`，然后生成轨迹
- `run_build_memory_with_rollout.sh`：自动启动 rollout server，然后生成 `memory_bank + memory.jsonl + dummy_memo.jsonl`
- `run_build_memory_pipeline.sh`：自动启动 rollout server，一次性生成 `memory_bank + memory.jsonl + dummy_memo.jsonl + embeddings + training_data`
- `build_embeddings.sh`：直接包装 `python -m baselines.MementoMath build-embeddings`
- `retrieve.sh`：直接包装 `python -m baselines.MementoMath retrieve`
- `run_build_training_data.sh`：直接包装 `python -m baselines.MementoMath build-training-data`
- `run_train_retriever.sh`：调用 `baselines.MementoMath` 内置训练器训练 parametric retriever
- `run_evolve_memory_with_rollout.sh`：自动启动 rollout server，增量更新 memory 库并导出新版本 case pool / dummy memory / embeddings

示例：

```bash
bash baselines/MementoMath/scripts/run_gen_traj.sh \
  --data-path /home/ycy/sdi/data/DeepMath-103K.jsonl \
  --start 0 --end 100 \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output baselines/MementoMath/outputs/traj.jsonl

bash baselines/MementoMath/scripts/run_build_memory_with_rollout.sh \
  --trajectories baselines/MementoMath/outputs/traj.jsonl \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output baselines/MementoMath/outputs/memory_bank.jsonl \
  --case-pool-output baselines/MementoMath/outputs/memory.jsonl \
  --dummy-memory-output baselines/MementoMath/outputs/dummy_memo.jsonl

bash baselines/MementoMath/scripts/build_embeddings.sh \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --output baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --backend hash

bash baselines/MementoMath/scripts/run_build_training_data.sh \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --embeddings baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --output baselines/MementoMath/outputs/training_data.jsonl

bash baselines/MementoMath/scripts/run_build_memory_pipeline.sh \
  --model ../models/Qwen3-4B-Instruct-2507

bash baselines/MementoMath/scripts/run_build_memory_pipeline.sh \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --train-retriever \
  --save-best \
  --fp16

bash baselines/MementoMath/scripts/run_train_retriever.sh \
  --train baselines/MementoMath/outputs/training_data.jsonl \
  --output-dir baselines/MementoMath/outputs/retriever_ckpts \
  --use-plan --save-best --fp16

bash baselines/MementoMath/scripts/run_evolve_memory_with_rollout.sh \
  --memory-bank baselines/MementoMath/outputs/memory_bank.jsonl \
  --trajectories baselines/MementoMath/outputs/new_traj.jsonl \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output-memory-bank baselines/MementoMath/outputs/memory_bank_v2.jsonl \
  --output-case-pool baselines/MementoMath/outputs/memory_v2.jsonl \
  --output-dummy-memory baselines/MementoMath/outputs/dummy_memo_v2.jsonl \
  --existing-embeddings baselines/MementoMath/outputs/memory_embeddings.jsonl \
  --output-embeddings baselines/MementoMath/outputs/memory_embeddings_v2.jsonl
```

## Data Contract

- 输入轨迹兼容 `SkillRL`：
  - `problem`
  - `topic`
  - `topic_key`
  - `student_response`
  - `is_correct`
  - `ground_truth`

- `memory_bank.jsonl` 字段：
  - `memory_id`
  - `query`
  - `topic`
  - `topic_key`
  - `status`
  - `reward`
  - `case_label`
  - `plan`
  - `plan_steps`
  - `takeaway`
  - `embedding_text`
  - `provenance`
  - `duplicate_count`

## 设计取向

- `success` 轨迹导出 `positive` case
- `failure` 轨迹导出 `negative` case
- 检索默认用 embedding 相似度，并对同 topic / 同 status 做轻量加分
- `evolve-memory` 支持增量加入新轨迹，并做去重合并
- retriever 训练直接复用原始 `Memento` 的 `train_memory_retriever.py`，不训练 LLM 本体，只训练 case-selection 模型
- 为了方便跨机器部署，训练与推理所需的 retriever 代码已内置到 `baselines/MementoMath`
