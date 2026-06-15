## Trace2Skill DeepMath 运行指南

本文档说明如何运行当前仓库中为 **DeepMath / SkillRL 数学轨迹**新增的 `Trace2Skill` baseline 适配流程。

当前适配入口是 `skill_evolver/run_deepmath_skill_evolution.py`，它会把 SkillRL 生成的数学题轨迹 JSONL 转成 Trace2Skill 原生的 error records，然后复用原始 Trace2Skill 的并行技能演化流程：

```text
SkillRL trajectory JSONL
        ↓
DeepMath adapter 转换为 Trace2Skill records
        ↓
ParallelSkillEvolver
        ↓
MAP → REDUCE → TRANSLATION → APPLY
        ↓
更新 released_skills/deepmath 技能目录
        ↓
可选：把演化后的 skill 注入 temp / greedy prompt 数据
```

## 1. 目录位置

请在 `Trace2Skill` baseline 根目录下运行本文命令：

```bash
cd /apdcephfs/cq/apdcephfs_cq10/share_1567347/share_info/cloudytang/Skill_Evo/baselines/Trace2Skill
```

关键文件如下：

```text
Trace2Skill/
├── skill_evolver/run_deepmath_skill_evolution.py   # DeepMath 适配入口
├── released_skills/deepmath/                       # 默认 DeepMath 技能目录
│   ├── SKILL.md
│   └── references/
└── outputs/                                        # 默认中间产物输出目录
```

## 2. 输入数据

默认输入轨迹文件为：

```text
../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl
```

该文件通常来自 `SkillRL` 的 rollout 结果。脚本会从每一行 JSON 中尝试读取以下字段：

- **题目**：`problem`、`question`、`raw_question`、`extra_info.problem`、`prompt`
- **标准答案**：`ground_truth`、`gt`、`answer`、`reward_model.ground_truth`、`extra_info.answer`
- **模型回答**：`student_response`、`response`、`completion`、`model_response`、`solution`
- **正确性标记**：`is_correct`
- **题目主题**：`topic`、`topic_key`、`extra_info.topic`

如果没有 `is_correct`，脚本会尝试从模型回答中抽取 `<answer>...</answer>` 或 `\boxed{...}`，再用 `mathruler.grader.grade_answer` 判断；如果没有安装 `mathruler`，则退化为字符串级别的简单比较。

## 3. 环境准备

建议先安装 Trace2Skill 原始运行依赖：

```bash
python -m pip install openai tqdm openpyxl requests diskcache
```

如果后续需要导出 parquet 格式的 prompt 数据，还需要安装：

```bash
python -m pip install pandas pyarrow
```

如果需要更准确地判断数学答案，建议安装当前环境可用的 `mathruler`：

```bash
python -m pip install mathruler
```

## 4. 模型 API 配置

脚本默认使用 OpenAI-compatible chat API。可通过环境变量或命令行参数配置：

```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=<optional_openai_compatible_endpoint>
```

也可以在命令里直接传入：

```bash
--api-key <your_api_key> \
--base-url <openai_compatible_endpoint>
```

如果使用本地 OpenAI-compatible 服务，常见写法是：

```bash
--api-key EMPTY \
--base-url http://localhost:8000/v1
```

## 5. 一键运行：演化技能并生成 prompt 数据

如果你希望一次完成 **轨迹转 records → 技能演化 → prompt 数据注入**，使用 `run-all`：

```bash
python -m skill_evolver.run_deepmath_skill_evolution run-all \
  --trajectories ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl \
  --skill-dir released_skills/deepmath \
  --model <model_name> \
  --api-key <your_api_key> \
  --base-url <openai_compatible_endpoint> \
  --batch-size 4 \
  --merge-batch-size 5 \
  --max-workers 4 \
  --max-records 0 \
  --include-success \
  --temp-input-jsonl ../SkillRL/outputs/temp_prompt_data.jsonl \
  --temp-output-jsonl outputs/deepmath_temp_trace2skill_prompt.jsonl \
  --greedy-input-jsonl ../SkillRL/outputs/greedy_prompt_data.jsonl \
  --greedy-output-jsonl outputs/deepmath_greedy_trace2skill_prompt.jsonl
```

其中：

- **`--trajectories`**：SkillRL rollout 轨迹 JSONL。
- **`--skill-dir`**：Trace2Skill 要更新的技能目录，默认是 `released_skills/deepmath`。
- **`--model`**：用于分析轨迹和修改技能的 LLM 名称。
- **`--max-records 0`**：表示不截断 records；调试时可以改成较小值，例如 `20`。
- **`--include-success`**：同时利用成功轨迹；不传时只使用失败或无法确认正确的轨迹。
- **`--temp-input-jsonl` / `--greedy-input-jsonl`**：需要注入技能的原始 prompt 数据。
- **`--temp-output-jsonl` / `--greedy-output-jsonl`**：注入 Trace2Skill 技能后的输出 prompt 数据。

如果你当前还没有 `temp_prompt_data.jsonl` 或 `greedy_prompt_data.jsonl`，可以先只跑技能演化，见下一节。

## 6. 只运行技能演化

只想从 SkillRL 轨迹里演化 `released_skills/deepmath`，运行：

```bash
python -m skill_evolver.run_deepmath_skill_evolution build-skill \
  --trajectories ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl \
  --skill-dir released_skills/deepmath \
  --model <model_name> \
  --api-key <your_api_key> \
  --base-url <openai_compatible_endpoint> \
  --batch-size 4 \
  --merge-batch-size 5 \
  --max-workers 4 \
  --max-records 100 \
  --include-success \
  --records-out outputs/deepmath_error_records.json \
  --intermediates-dir outputs/deepmath_parallel_output \
  --parse-failure-dir parse_failures_deepmath \
  --prompt generic \
  --patch-pipeline json
```

运行后主要产物包括：

```text
released_skills/deepmath/              # 被更新的技能目录
outputs/deepmath_error_records.json    # 由轨迹转换出的 Trace2Skill records
outputs/deepmath_parallel_output/      # MAP / REDUCE / TRANSLATION 中间结果
parse_failures_deepmath/               # patch 解析失败样本，若存在则用于排查
```

建议调试时先设置较小的 `--max-records`，确认链路可跑通后再放大数据量。

## 7. 只生成注入技能后的 prompt 数据

如果 `released_skills/deepmath` 已经演化完成，只需要把技能内容注入到 DeepMath prompt 数据中，使用 `prepare-prompt-data`：

```bash
python -m skill_evolver.run_deepmath_skill_evolution prepare-prompt-data \
  --input-jsonl ../SkillRL/outputs/temp_prompt_data.jsonl \
  --skill-dir released_skills/deepmath \
  --output-jsonl outputs/deepmath_temp_trace2skill_prompt.jsonl \
  --output-parquet outputs/deepmath_temp_trace2skill_prompt.parquet \
  --data-source Trace2SkillMath \
  --start 0
```

如果不需要 parquet，可以去掉 `--output-parquet`：

```bash
python -m skill_evolver.run_deepmath_skill_evolution prepare-prompt-data \
  --input-jsonl ../SkillRL/outputs/greedy_prompt_data.jsonl \
  --skill-dir released_skills/deepmath \
  --output-jsonl outputs/deepmath_greedy_trace2skill_prompt.jsonl \
  --data-source Trace2SkillMath
```

输出 JSONL 每行会包含：

- **`problem`**：原始数学题。
- **`ground_truth`**：标准答案。
- **`prompt`**：注入 Trace2Skill 技能后的用户 prompt。
- **`reward_model.ground_truth`**：用于后续评测或训练的答案字段。
- **`extra_info.trace2skill_memory_dir`**：使用的技能目录。
- **`extra_info.trace2skill_memory_chars`**：注入技能文本长度。

## 8. 常用参数说明

### 技能演化相关

- **`--batch-size`**：MAP 阶段每个分析 batch 的 records 数量，默认 `4`。
- **`--merge-batch-size`**：REDUCE 阶段每轮合并的 patch 数量，默认 `5`。
- **`--max-workers`**：并行 worker 数，默认 `4`。
- **`--max-merge-levels`**：层级合并的最大轮数，默认 `5`。
- **`--temperature`**：技能演化 LLM 调用温度，默认 `0.3`。
- **`--max-tokens`**：LLM 单次响应最大 token 数，默认不显式限制。
- **`--max-skill-lines`**：技能文件最大行数，默认 `500`。
- **`--prompt`**：技能演化 prompt 变体，可选 `skill`、`generic`、`patterns`、`patterns_generic`，默认 `generic`。
- **`--patch-pipeline`**：patch 格式，可选 `json` 或 `markdown`，默认 `json`。
- **`--skip-translation`**：跳过 TRANSLATION 阶段，通常不建议开启。
- **`--dry-run`**：只生成 patch，不真正写入技能目录。
- **`--verbose`**：输出更详细日志。

### 数据截断与筛选

- **`--max-records`**：最多使用多少条转换后的 records；`0` 表示不限制。
- **`--include-success`**：是否把成功轨迹也转成可学习的经验记录。
- **`--start` / `--end`**：生成 prompt 数据时截取输入 JSONL 的行号范围。

### 模型客户端

- **`--llm-client openai`**：默认，使用 OpenAI-compatible 接口。
- **`--llm-client api_chat`**：使用 `ApiChatClient`。
- **`--api-chat-config`**：`api_chat` 模式配置文件，默认 `config/llm_api.json`。
- **`--cache-path`**：可选的 LLM 请求缓存路径。

## 9. 推荐调试流程

第一次跑建议按下面顺序来：

```bash
# 1. 先用少量 records 检查技能演化链路
python -m skill_evolver.run_deepmath_skill_evolution build-skill \
  --trajectories ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl \
  --skill-dir released_skills/deepmath \
  --model <model_name> \
  --api-key <your_api_key> \
  --base-url <openai_compatible_endpoint> \
  --max-records 10 \
  --max-workers 2 \
  --verbose

# 2. 确认技能目录被正确更新
cat released_skills/deepmath/SKILL.md

# 3. 生成一小段注入后的 prompt 数据
python -m skill_evolver.run_deepmath_skill_evolution prepare-prompt-data \
  --input-jsonl ../SkillRL/outputs/temp_prompt_data.jsonl \
  --skill-dir released_skills/deepmath \
  --output-jsonl outputs/debug_trace2skill_prompt.jsonl \
  --start 0 \
  --end 5
```

确认无误后，再把 `--max-records` 调大或设为 `0`，并增加 `--max-workers`。

## 10. 与 SkillOpt baseline 的区别

两个 baseline 都会使用 SkillRL 相关数据，但方式不同：

- **Trace2Skill**：直接利用已有 rollout 轨迹中的 `student_response`、`is_correct`、`ground_truth` 等信息，把轨迹转成 error records，再离线演化技能目录。
- **SkillOpt**：主要把同一个 JSONL 当作数学题任务池，重新 rollout 当前 skill，再根据新 rollout 的结果做在线式技能优化。

因此，Trace2Skill 更像是 **从历史轨迹中蒸馏经验**；SkillOpt 更像是 **围绕当前技能持续试题、反思和更新**。

## 11. 常见问题

### 找不到 `skill_evolver` 模块

请确认你在 `Trace2Skill` 根目录下运行命令：

```bash
pwd
```

应输出类似：

```text
/apdcephfs/cq/apdcephfs_cq10/share_1567347/share_info/cloudytang/Skill_Evo/baselines/Trace2Skill
```

### 没有生成 records

如果出现：

```text
No Trace2Skill error records were produced from trajectories.
```

通常说明输入 JSONL 中缺少题目或模型回答字段。请检查每行是否至少能解析出：

- 题目：`problem` / `question` / `prompt`
- 模型回答：`student_response` / `response` / `completion` / `model_response`

### parquet 写入失败

如果 `--output-parquet` 报错，通常是缺少依赖：

```bash
python -m pip install pandas pyarrow
```

不需要 parquet 时，直接去掉 `--output-parquet` 即可。

### 不想覆盖默认技能目录

可以先复制一份技能目录，再对副本演化：

```bash
cp -r released_skills/deepmath outputs/deepmath_skill_work

python -m skill_evolver.run_deepmath_skill_evolution build-skill \
  --trajectories ../SkillRL/outputs/trajectories_from_merged_v1_v2.jsonl \
  --skill-dir outputs/deepmath_skill_work \
  --model <model_name> \
  --api-key <your_api_key> \
  --base-url <openai_compatible_endpoint>
```

这样可以保留 `released_skills/deepmath` 的初始版本，方便对比。
