# Skill Memory System API 文档

## 概述

Skill Memory 系统由两个独立的 HTTP 服务组成：
- **retriever_server**（端口 8766）：独立 embedding 检索服务，基于 vLLM 加载 embedding 模型
- **memory_server**（端口 8765）：Skill 存储与管理服务，提供 CRUD、检索、评估更新等功能

---

## Retriever Server（独立 Embedding 检索）

**基础 URL**: `http://127.0.0.1:8766`

### 1. GET `/health`

检查服务状态。

**请求**：无请求体

**响应**：
```json
{
  "ok": true,
  "model_loaded": true,
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "idle_timeout": 300.0,
  "idle_remaining": 299.5
}
```

**说明**：
- `idle_remaining`：距离自动卸载还剩多少秒（空闲计时器）

---

### 2. POST `/encode`

对文本列表进行 embedding 编码。

**请求**：
```json
{
  "texts": ["文本1", "文本2", "文本3"],
  "is_query": false
}
```

**字段说明**：
- `texts`（必需）：字符串列表，非空
- `is_query`（可选，默认 false）：
  - `true` - 使用 query prompt（用于问题文本）
  - `false` - 使用 document 编码（用于文档文本）

**响应**：
```json
{
  "ok": true,
  "embeddings": [
    [0.1, 0.2, -0.15, ...],
    [0.05, 0.25, -0.1, ...],
    [0.12, 0.18, -0.2, ...]
  ]
}
```

**说明**：
- `embeddings`：浮点向量列表，每个向量维度根据模型决定（通常 384 或 768）

---

### 3. POST `/rank`

使用问题对候选文本列表进行排序。

**请求**：
```json
{
  "question": "如何解决二次方程？",
  "candidates": [
    {"problem_type": "代数方程求解", "utility": 0.8},
    {"problem_type": "多项式因式分解", "utility": 0.5},
    {"problem_type": "三角函数化简", "utility": 0.3}
  ],
  "mode": "embedding",
  "retrieve_lambda": 0.5,
  "top_k": 2
}
```

**字段说明**：
- `question`（必需）：问题文本
- `candidates`（必需）：候选对象列表，每个对象包含：
  - `problem_type`（必需）：文本描述
  - `utility`（必需）：浮点数 [0.0, 1.0]，表示该候选的质量分数
- `mode`（可选，默认 "embedding"）：
  - `"embedding"` - 仅按相似度排序
  - `"hybrid"` - 混合排序：`score = (1-λ)*similarity + λ*utility`
- `retrieve_lambda`（可选，默认 0.5）：hybrid 模式中 utility 的权重（仅在 mode="hybrid" 时生效）
- `top_k`（可选）：返回排序后的前 top_k 个索引

**响应**：
```json
{
  "ok": true,
  "ranked_indices": [0, 1, 2]
}
```

**说明**：
- `ranked_indices`：排序后的候选索引列表（降序排列，最相关的在前）
- 索引对应 request 中 `candidates` 列表的位置

---

## Memory Server（Skill 存储与管理）

**基础 URL**: `http://127.0.0.1:8765`

### 1. GET `/health`

检查 memory_server 和 retriever_server 状态。

**请求**：无请求体

**响应**：
```json
{
  "ok": true,
  "retriever_ready": true,
  "retriever_url": "http://127.0.0.1:8766",
  "current_size": 150,
  "max_capacity": 10000
}
```

---

### 2. POST `/retrieve`

根据问题检索相关 skill。

**请求**：
```json
{
  "question": "如何解决自定义二元运算的嵌套问题？",
  "top_k": 5
}
```

**字段说明**：
- `question`（必需）：问题文本
- `top_k`（可选）：返回前 top_k 个结果

**响应**：
```json
{
  "ok": true,
  "skills": [
    {
      "id": "0",
      "skill name": "Evaluate nested custom operations",
      "problem type": "custom binary operations with associativity testing",
      "key insight": "Custom operations often lack associativity...",
      "method": "1. Apply operation... 2. Compute... 3. Subtract..."
    },
    {
      "id": "3",
      "skill name": "Solve equations via substitution",
      "problem type": "algebra system of equations",
      "key insight": "Symmetric systems often admit x=y solutions...",
      "method": "1. Try x=y. 2. Substitute... 3. Solve..."
    }
  ],
  "count": 2
}
```

**说明**：
- 返回的 skill 包含 5 个字段：`id` + 4 个内容字段（不含 utility 等其他内部字段）
- `id` 用于后续 `/update` 调用，可唯一标识 skill，避免 skill_name 重复时的歧义
- 按相关性排序（embedding 相似度或 hybrid 混合分数）

---

### 3. POST `/add`

添加新 skill 到内存。

**请求**：
```json
{
  "skill name": "Quadratic formula application",
  "problem type": "algebra quadratic equations",
  "key insight": "Apply the quadratic formula directly when factoring is difficult",
  "method": "1. Identify a,b,c. 2. Apply x=(-b±√(b²-4ac))/(2a). 3. Simplify.",
  "skill_from": "success_rollout",
  "problem": "Solve 2x^2 - 3x - 2 = 0.",
  "reward": 0.85
}
```

**字段说明**：
- `skill name`（必需）：技能名称，字符串
- `problem type`（必需）：问题类型，字符串
- `key insight`（必需）：关键洞察，字符串
- `method`（必需）：解决方法，字符串
- `skill_from`（必需）：来源，通常为 "success_rollout" 或 "fail_rollout"
- `problem`（必需）：具体问题描述，字符串
- `reward`（必需）：浮点数，表示 skill 的初始价值
- 注：`id` 由服务端自动分配，无需提供

**响应**：
```json
{
  "ok": true,
  "id": "42",
  "zone": "main",
  "evicted_id": null,
  "persist_path": "/path/to/skills_memory.jsonl"
}
```

**说明**：
- `id`：服务端分配的自增整数字符串
- `zone`：添加到的区域 - "main"（主库）或 "warning"（警告区，主库满时）
- `evicted_id`：如果警告区满被淘汰的 skill id（可能为 null）
- `persist_path`：持久化文件路径

---

### 4. POST `/update`

批量更新 skill 的效用值和使用计数。

**请求**：
```json
{
  "skills": [
    {
      "id": "42",
      "is_success": true,
      "reward": 0.9
    },
    {
      "id": "7",
      "is_success": false,
      "reward": 0.3
    }
  ]
}
```

**字段说明**（单条更新项）：
- `id`（必需）：skill 的唯一 id，由 `/add` 返回或 `/retrieve` 结果中获取
- `is_success`（必需）：布尔值，true 表示使用成功，false 表示失败
- `reward`（必需）：浮点数，本次使用的奖励值

**响应**：
```json
{
  "ok": true,
  "results": [
    {
      "index": 0,
      "id": "42",
      "ok": true,
      "zone": "main",
      "action": "updated",
      "utility_before": 0.5,
      "utility_after": 0.63,
      "promoted_from_warn_id": null,
      "demoted_to_warn_id": null,
      "evicted_warn_id": null
    },
    {
      "index": 1,
      "id": "7",
      "ok": true,
      "zone": "main",
      "action": "no_change",
      "utility_before": 0.6,
      "utility_after": 0.6
    }
  ]
}
```

**响应字段说明**：
- `index`：该条在请求数组中的位置
- `id`：对应的 skill id
- `zone`：该 skill 所在区域（"main" 或 "warning"）
- `action`：执行的动作：
  - `"updated"` - utility 已更新
  - `"no_change"` - utility 不变（reward < τ 时）
  - `"removed"` - skill 被彻底删除（警告区失败时）
  - `"promoted"` - 从警告区晋升到主库
  - `"stayed"` - 在警告区但 utility 已更新
- `utility_before` / `utility_after`：更新前后的效用值
- `promoted_from_warn_id`：晋升时该 skill 的 id
- `demoted_to_warn_id`：晋升时被替换到警告区的 skill id
- `evicted_warn_id`：晋升时警告区满而被淘汰的 skill id

**内部机制**：
- 计算新 utility：EMA 更新，`new_util = old_util + α * sign(is_success) * max(0, |reward| - τ)`
- 主库满时新 skill 进入警告区，警告区满时淘汰最低 utility 的 skill
- 警告区 skill 失败时彻底删除；成功且 utility 足够高时可晋升回主库

---

### 5. POST `/manage`

管理接口，支持多种操作。

#### 5.1 GET Status

获取全局状态。

**请求**：
```json
{
  "action": "status"
}
```

**响应**：
```json
{
  "ok": true,
  "current_size": 150,
  "max_capacity": 10000,
  "is_full": false,
  "warn_size": 5,
  "warn_capacity": 200,
  "retrieve_mode": "embedding",
  "retrieve_lambda": 0.5,
  "retriever_url": "http://127.0.0.1:8766"
}
```

#### 5.2 GET Warning Zone Status

获取警告区详细信息。

**请求**：
```json
{
  "action": "warn_status"
}
```

**响应**：
```json
{
  "ok": true,
  "warn_size": 5,
  "warn_capacity": 200,
  "skills": [
    {
      "skill name": "...",
      "problem type": "...",
      "key insight": "...",
      "method": "...",
      "id": "42",
      "skill_from": "...",
      "problem": "...",
      "utility": 0.3,
      "skill_usage_success": 2,
      "skill_usage_failure": 3
    }
  ]
}
```

#### 5.3 GET Single Skill

按 id 获取单条 skill 完整信息。

**请求**：
```json
{
  "action": "get",
  "id": "42"
}
```

**响应**：
```json
{
  "ok": true,
  "skill": {
    "skill name": "Quadratic formula application",
    "id": "42",
    "utility": 0.85,
    "skill_usage_success": 10,
    "skill_usage_failure": 2,
    ...
  }
}
```

#### 5.4 REMOVE Single Skill

删除指定 id 的 skill。

**请求**：
```json
{
  "action": "remove",
  "id": "42"
}
```

**响应**：
```json
{
  "ok": true,
  "removed": true
}
```

#### 5.5 LIST All IDs

列出所有 skill id。

**请求**：
```json
{
  "action": "list_ids"
}
```

**响应**：
```json
{
  "ok": true,
  "ids": ["0", "1", "2", "3", ...]
}
```

---

## 数据字段汇总

### 完整 Skill 对象
```json
{
  "id": "42",
  "skill name": "Quadratic formula application",
  "problem type": "algebra quadratic equations",
  "key insight": "Apply the quadratic formula directly",
  "method": "1. Identify a,b,c. 2. Apply...",
  "skill_from": "success_rollout",
  "problem": "Solve 2x^2 - 3x - 2 = 0.",
  "utility": 0.85,
  "skill_usage_success": 10,
  "skill_usage_failure": 2
}
```

### 检索返回的 Skill（简化版）
```json
{
  "skill name": "Quadratic formula application",
  "problem type": "algebra quadratic equations",
  "key insight": "Apply the quadratic formula directly",
  "method": "1. Identify a,b,c. 2. Apply..."
}
```

---

## 错误处理

所有响应都包含 `"ok"` 字段：
- `"ok": true` - 请求成功
- `"ok": false` - 请求失败，查看 `"error"` 字段获取错误信息

常见 HTTP 状态码：
- `200 OK` - 请求成功
- `400 Bad Request` - 请求体格式错误或缺少必需字段
- `404 Not Found` - 资源不存在
- `409 Conflict` - ID 冲突或业务逻辑冲突
- `500 Internal Server Error` - 服务器内部错误
- `507 Insufficient Storage` - 内存已满

---

## 启动命令

### Retriever Server
```bash
bash skill_zero/memory_manager/start_retriever_server.sh
```

### Memory Server
```bash
bash skill_zero/memory_manager/start_memory_server.sh
```

或者手动启动带自定义参数：
```bash
# Memory Server（小容量用于测试）
python -m skill_zero.memory_manager.memory_server \
    --max-capacity 3 --warn-capacity 2 \
    --retriever-url http://127.0.0.1:8766

# Retriever Server
python -m skill_zero.memory_manager.retriever_server \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --embedding-device cuda:0 \
    --idle-timeout 300
```
