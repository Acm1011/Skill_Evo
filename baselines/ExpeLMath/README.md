# baselines/ExpeLMath

数学版 ExpeL baseline，面向 `DeepMath` / `SkillRL` 轨迹，核心聚焦：

- 从成功/失败轨迹抽取 ExpeL-style reusable insights
- 将 insight 同时保存为 `raw_rule` 与结构化 `memory_items`
- 用 embedding + topic/rule-type 重排检索 memory
- 在数学推理时把 memory 注入 prompt

## CLI

```bash
export PYTHONPATH=/path/to/Skill_Evo:$PYTHONPATH

python -m baselines.ExpeLMath gen-traj \
  --data-path /path/to/DeepMath-103K.jsonl \
  --start 0 --end 100 \
  --output baselines/ExpeLMath/outputs/traj.jsonl \
  --server-urls http://127.0.0.1:8760 http://127.0.0.1:8761 http://127.0.0.1:8762 http://127.0.0.1:8763

python -m baselines.ExpeLMath build-memory \
  --teacher-backend rollout \
  --trajectories baselines/ExpeLMath/outputs/traj.jsonl \
  --output baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --rollout-host 127.0.0.1 --rollout-base-port 8760 --rollout-n-servers 4

python -m baselines.ExpeLMath build-memory \
  --teacher-backend chat \
  --trajectories baselines/ExpeLMath/outputs/traj.jsonl \
  --output baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --teacher-model your-teacher-model

python -m baselines.ExpeLMath build-embeddings \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --output baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --backend hash

python -m baselines.ExpeLMath retrieve \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --embeddings baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --question "Solve x + 2 = 5" \
  --topic "Math->Algebra" \
  --top-k 3

python -m baselines.ExpeLMath eval \
  --deepmath-jsonl /path/to/DeepMath-103K.jsonl \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --embeddings baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --start 0 --end 100 \
  --output baselines/ExpeLMath/outputs/eval.jsonl \
  --server-urls http://127.0.0.1:8760 http://127.0.0.1:8761 http://127.0.0.1:8762 http://127.0.0.1:8763

python -m baselines.ExpeLMath evolve-memory \
  --teacher-backend rollout \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --trajectories baselines/ExpeLMath/outputs/new_traj.jsonl \
  --output-memory-bank baselines/ExpeLMath/outputs/memory_bank_v2.jsonl \
  --existing-embeddings baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --output-embeddings baselines/ExpeLMath/outputs/memory_embeddings_v2.jsonl \
  --rollout-host 127.0.0.1 --rollout-base-port 8760 --rollout-n-servers 4
```

## 什么时候需要 model 和 localhost

- `gen-traj`：需要学生 rollout server，所以要么传 `--server-urls http://127.0.0.1:8760 ...`，要么用下面的 `scripts/run_gen_traj.sh` 自动起服务。
- `build-memory` / `evolve-memory`：
  - `--teacher-backend rollout`：需要 rollout server，所以需要 `localhost + 端口`。
  - `--teacher-backend chat`：不需要 rollout server，但需要 `--teacher-base-url` 和 `--teacher-model`。
- `eval`：需要学生 rollout server 生成解题结果，所以也需要 `--server-urls` 或脚本自动起服务。
- `build-embeddings`：
  - `--backend hash`：不需要任何模型服务。
  - `--backend openai`：需要 `--embed-base-url` 和 `--embed-model`。
- `retrieve`：默认只读已有 embedding 文件并做本地相似度排序，不需要外部 retriever server。

## evolve-memory 是做什么

`evolve-memory` 不是“重头再建一遍 memory bank”，而是：

- 读取已有 `memory_bank.jsonl`
- 用新轨迹 `new_traj.jsonl` 再生成一批新的 memory
- 用 `similarity_threshold` 和 `memory_type` 规则做去重合并
- 把重复样本并到已有 memory 的 `provenance/duplicate_count`
- 可选同步刷新 `memory_embeddings.jsonl`

适合的场景是你后续又 rollout 了一批新题或新版本轨迹，不想把旧 bank 全丢掉重建。

## 直接可执行脚本

脚本目录：[scripts/](scripts)

- `run_gen_traj.sh`：自动启动 `skill_src/Zero/start_rollout_servers.sh`，然后生成轨迹
- `run_build_memory_with_rollout.sh`：自动启动 rollout server，然后用 rollout backend 建 memory
- `build_embeddings.sh`：直接包装 `python -m baselines.ExpeLMath build-embeddings`
- `retrieve.sh`：直接包装 `python -m baselines.ExpeLMath retrieve`
- `run_eval_with_rollout.sh`：自动启动 rollout server，然后执行带 memory 的评测
- `run_evolve_memory_with_rollout.sh`：自动启动 rollout server，然后做增量 memory 演化

示例：

```bash
bash baselines/ExpeLMath/scripts/run_gen_traj.sh \
  --data-path /home/ycy/sdi/data/DeepMath-103K.jsonl \
  --start 0 --end 100 \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output baselines/ExpeLMath/outputs/traj.jsonl

bash baselines/ExpeLMath/scripts/run_build_memory_with_rollout.sh \
  --trajectories baselines/ExpeLMath/outputs/traj.jsonl \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output baselines/ExpeLMath/outputs/memory_bank.jsonl

bash baselines/ExpeLMath/scripts/build_embeddings.sh \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --output baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --backend hash

bash baselines/ExpeLMath/scripts/retrieve.sh \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --embeddings baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --question "Solve x + 2 = 5" \
  --topic "Math->Algebra" \
  --top-k 3

bash baselines/ExpeLMath/scripts/run_eval_with_rollout.sh \
  --deepmath-jsonl /home/ycy/sdi/data/DeepMath-103K.jsonl \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --embeddings baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --start 0 --end 100 \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output baselines/ExpeLMath/outputs/eval.jsonl

bash baselines/ExpeLMath/scripts/run_evolve_memory_with_rollout.sh \
  --memory-bank baselines/ExpeLMath/outputs/memory_bank.jsonl \
  --trajectories baselines/ExpeLMath/outputs/new_traj.jsonl \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --output-memory-bank baselines/ExpeLMath/outputs/memory_bank_v2.jsonl \
  --existing-embeddings baselines/ExpeLMath/outputs/memory_embeddings.jsonl \
  --output-embeddings baselines/ExpeLMath/outputs/memory_embeddings_v2.jsonl
```

## 关于 `start_retriever_server.sh`

当前 ExpeLMath 的 `retrieve` 走的是本地 `embedding_rows + cosine similarity` 检索，不会调用 `skill_src/Zero/start_retriever_server.sh`。也就是说：

- 你现在跑 ExpeLMath，不需要启动 retriever server
- 只需要在 rollout 相关步骤启动 `skill_src/Zero/start_rollout_servers.sh`
- 如果后面你想把 ExpeLMath 改成走外部 retriever HTTP 排序，那时再接 `start_retriever_server.sh` 才有意义

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
- `success_rule`: 成功轨迹归纳出的正向规则
- `failure_rule`: 多条失败轨迹归纳出的避免性规则

## Grouping Strategy

- 按题分组后，每类 memory 最多只取两条轨迹
- mixed 题：取 1 条成功 + 1 条失败生成 `compare_rule`
- mixed 且失败数至少 2：额外取 2 条失败生成 `failure_rule`
- 纯成功题：取 2 条成功生成 `success_rule`
- 纯失败题：取 2 条失败生成 `failure_rule`
