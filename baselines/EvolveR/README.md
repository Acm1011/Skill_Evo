# baselines/EvolveR — 数学任务 + 体验库多轮（无外部 Wiki）

本目录包含 **vendored** 的 [EvolveR/verl](../../EvolveR/verl) 副本与 **可改** 的 `evolver/`，在 **不修改** `verl` 源码的前提下，将任务改为数学（DeepMath 风格数据）、**保留** `<search_experience>` 多轮与 Milvus 体验库、**默认不启用** 外部知识库 `<search_knowledge>`（`EVOLVER_KNOWLEDGE_SEARCH=0`）。

与 [baselines/SkillRL](../SkillRL) 的对照：

| 项 | SkillRL | 本目录 |
|----|---------|--------|
| 训练栈 | 仓库 [SkillRL/verl](../../SkillRL/verl) | 本目录下 `verl/` + `evolver/`，不合并 SkillRL |
| 多轮 + 体验库 | 无 | 有（`run_llm_loop`、VDB） |
| 数据列 | `prompt` + `reward_model.ground_truth` | 相同，并需 `question` / `golden_answers` / `ability`（`build_math_parquet.py` 已写） |

## 目录

- `verl/`：自 EvolveR 拷贝的 **不要手改**（与上游 diff 用）。
- `evolver/`：可改；数学奖励、`no_wiki` 格式奖励、无 KB 的 rollout 等。
- `config/ppo_trainer_math.yaml`：数学 system prompt、默认奖励权重等；入口见 `run_ppo_math.py`。
- `run_ppo_math.py`：使用 **本目录** `config/`，`import verl.trainer.main_ppo.main_task` 跑训练。

## 依赖

```bash
pip install -r baselines/EvolveR/requirements.txt
# 以及与你 GPU 匹配的 torch / vllm
```

将 **`baselines/EvolveR`** 加入 `PYTHONPATH`（脚本已设置）。

## 环境变量（节选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `EVOLVER_KNOWLEDGE_SEARCH` | `0` | `1` 时对 `<search_knowledge>` 发 HTTP 检索 |
| `EVOLVER_QA_OUTCOME` | `0` | `1` 时用旧问答 EM 而非数学等价 |
| `EVOLVER_REQUIRE_KNOWLEDGE_SEARCH` | 关闭 | 设为 `1` 时 format 走「必须 search_knowledge」的 legacy 逻辑 |

## 数据：DeepMath → parquet

```bash
export PYTHONPATH="/path/to/Skill_Evo:${PYTHONPATH}"
export DEEPMATH_JSONL=/data/DeepMath-103K.jsonl
export OUT_TRAIN=./outputs/deepmath_rl_train.parquet
export END=10000
bash baselines/EvolveR/scripts/prepare_math_parquet.sh
```

可选验证集：`export VAL_RATIO=0.05 OUT_VAL=./outputs/deepmath_rl_val.parquet`。

## 服务（训练前）

1. **Embedding（BGE 等，OpenAI 兼容）**：与 `experience.embedding_api_url` 一致，例如 `scripts/vllm_embedding.sh`（默认 `8081`）。
2. **体验 VDB（Milvus）**：`bash baselines/EvolveR/scripts/start_milvus_vdb.sh`（`db_server.py` 默认端口 **8007**；`run_rl` 中可按需设置 `VDB_SERVER_URL`）。
3. **Wiki 检索**：本 baseline 不需要。若将 `EVOLVER_KNOWLEDGE_SEARCH=1`，可起 `scripts/retriever_stub.sh` 或自建检索。

## 训练

```bash
export PYTHONPATH="/path/to/Skill_Evo:/path/to/Skill_Evo/baselines/EvolveR:${PYTHONPATH}"
export TRAIN_FILE=.../deepmath_rl_train.parquet
export VAL_FILE=.../deepmath_rl_val.parquet
export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
export EMBEDDING_API_URL=http://127.0.0.1:8081/v1
export VDB_SERVER_URL=http://127.0.0.1:8007
bash baselines/EvolveR/scripts/run_rl_grpo_evolve_math.sh
```

更底层可直接：`cd baselines/EvolveR && python run_ppo_math.py data.train_files=...`（Hydra 覆盖）。

## 与 SkillRL 的 parquet 差异

`build_math_parquet.py` 在 SkillRL 相同字段基础上，为 EvolveR/verl 增加：`question`（= 题目文本）、`golden_answers`（列表）、`ability: math`，以便 `ray_trainer` 与 experience 存轨迹。
