#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
memory_server + retriever_server 功能测试脚本。

使用方式：
    # 建议以小容量启动，以便触发警告区逻辑
    # 终端1：启动 retriever_server（正常启动）
    bash start_retriever_server.sh
    # 终端2：启动 memory_server，设置小容量
    python -m memory_server \
        --max-capacity 3 --warn-capacity 2 \
        --retriever-url http://127.0.0.1:8766 \
        --persist-path /tmp/test_skills.jsonl
    # 终端3：运行测试
    python -m test_server

结果保存到 skill_zero/memory_manager/test_results/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("请先安装 requests：pip install requests")

MEMORY_URL    = "http://127.0.0.1:8765"
RETRIEVER_URL = "http://127.0.0.1:8766"
OUTPUT_DIR    = Path(__file__).parent / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def save(name: str, data: dict | list) -> Path:
    p = OUTPUT_DIR / f"{name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [saved] {p}")
    return p


def get(base: str, endpoint: str, timeout: int = 15) -> dict:
    return requests.get(f"{base}{endpoint}", timeout=timeout).json()


def post(base: str, endpoint: str, payload: dict, timeout: int = 60) -> dict:
    return requests.post(f"{base}{endpoint}", json=payload, timeout=timeout).json()


def section(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def wait_ready(url: str, label: str, max_wait: int = 120) -> None:
    print(f"等待 {label} 就绪（最多 {max_wait}s）...")
    for i in range(max_wait):
        try:
            requests.get(f"{url}/health", timeout=3)
            print(f"  {label} 已响应。")
            return
        except Exception:
            if i == max_wait - 1:
                sys.exit(f"{label} {max_wait}s 内未响应，请先启动对应 server。")
            time.sleep(1)
            print(f"  等待中... ({i+1}s)", end="\r")


def get_warn_status() -> dict:
    return post(MEMORY_URL, "/manage", {"action": "warn_status"})


def get_main_status() -> dict:
    return post(MEMORY_URL, "/manage", {"action": "status"})


def print_zone_summary(label: str) -> dict:
    st = get_main_status()
    ws = get_warn_status()
    print(f"  [{label}] 主库: {st.get('current_size')}/{st.get('max_capacity')}  "
          f"警告区: {ws.get('warn_size')}/{ws.get('warn_capacity')}")
    warn_names = [it.get("skill name") for it in ws.get("skills", [])]
    if warn_names:
        print(f"           警告区内容: {warn_names}")
    return {"main_status": st, "warn_status": ws}


# ─── 测试数据 ─────────────────────────────────────────────────────────────────
# 设计5条skill，reward依次不同，便于观察主库/警告区的utility排序和晋升/淘汰
# 主库容量=3，警告区容量=2：前3条进主库，第4、5条进警告区

SKILLS_TO_ADD = [
    {
        "skill name": "Evaluate nested custom operations",
        "problem type": "custom binary operations with associativity testing",
        "key insight": "Custom operations often lack associativity; compute inner expressions first.",
        "method": "1. Apply operation to innermost parentheses. 2. Compute each nested result. 3. Subtract.",
        "skill_from": "success_rollout",
        "problem": "The operation $\\otimes$ is defined by $a\\otimes b=a^2/b$. Find $[(1\\otimes2)\\otimes3]-[1\\otimes(2\\otimes3)]$.",
        "reward": 0.8,   # 进主库（第1条）
    },
    {
        "skill name": "Determine grid layout for square window",
        "problem type": "geometry with constraints and ratios",
        "key insight": "Total height equals width for a square; borders contribute fixed widths.",
        "method": "1) List arrangements. 2) Compute total dimensions. 3) Set height=width and solve.",
        "skill_from": "success_rollout",
        "problem": "Doug constructs a square window using 8 panes. Find side length.",
        "reward": 0.5,   # 进主库（第2条）
    },
    {
        "skill name": "Pattern recognition in periodic interpolation",
        "problem type": "Polynomial interpolation with periodic value constraints",
        "key insight": "A polynomial matching a periodic pattern can be expressed via root terms.",
        "method": "1. Identify periodic values. 2. Write as roots product. 3. Evaluate at target.",
        "skill_from": "fail_rollout",
        "problem": "Polynomial P(x) of degree 3n; P(3n+1)=730. Find n.",
        "reward": 0.1,   # 进主库（第3条，utility最低，是晋升的候选替换对象）
    },
    {
        "skill name": "Counting arrangements with forbidden adjacency",
        "problem type": "combinatorics with constraints",
        "key insight": "Use complementary counting: total minus those where forbidden pairs are adjacent.",
        "method": "1. Total permutations. 2. Treat forbidden pair as block. 3. Subtract.",
        "skill_from": "success_rollout",
        "problem": "5 people sit in a row; two specific people must not be adjacent. Count arrangements.",
        "reward": 0.9,   # 主库满→进警告区（第4条）
    },
    {
        "skill name": "Solve equations via substitution and symmetry",
        "problem type": "algebra system of equations",
        "key insight": "Symmetric systems admit x=y solutions; substitution reduces degrees of freedom.",
        "method": "1. Try x=y. 2. Substitute. 3. Solve reduced equation. 4. Verify.",
        "skill_from": "success_rollout",
        "problem": "Solve: x+y=5, x^2+y^2=13.",
        "reward": 0.7,   # 警告区还有1位→进警告区（第5条）
    },
]

# reward 低于 tau=0.2，effective_reward=0，不触发更新
REWARD_BELOW_TAU = 0.1


# ─── STEP 0：健康检查 ─────────────────────────────────────────────────────────

def check_health() -> None:
    section("STEP 0 — 健康检查")

    rh = get(RETRIEVER_URL, "/health")
    print(f"  retriever_server: model_loaded={rh.get('model_loaded')}, "
          f"idle_timeout={rh.get('idle_timeout')}s, idle_remaining={rh.get('idle_remaining')}s")
    save("00a_retriever_health", rh)
    assert rh.get("model_loaded"), f"retriever_server 模型未加载: {rh}"

    mh = get(MEMORY_URL, "/health")
    print(f"  memory_server:    retriever_ready={mh.get('retriever_ready')}, "
          f"size={mh.get('current_size')}/{mh.get('max_capacity')}")
    save("00b_memory_health", mh)
    assert mh.get("ok"), f"memory_server 未就绪: {mh}"
    assert mh.get("retriever_ready"), "retriever 未就绪，请确认 retriever_server 已启动"
    print("  [OK] 两个 server 均健康")


# ─── STEP 1：retriever_server 直接测试 ────────────────────────────────────────

def test_retriever_directly() -> None:
    section("STEP 1 — 直接测试 retriever_server 接口")

    section("STEP 1a — /encode (is_query=true)")
    enc = post(RETRIEVER_URL, "/encode", {
        "texts": ["custom binary operation", "geometry window panes"],
        "is_query": True,
    })
    print(f"  ok={enc.get('ok')}, count={len(enc.get('embeddings', []))}")
    if enc.get("embeddings"):
        dim = len(enc["embeddings"][0])
        print(f"  向量维度: {dim}")
        save("01a_encode_query", {
            "ok": enc["ok"],
            "embeddings_shape": [len(enc["embeddings"]), dim],
            "embeddings_preview": [v[:8] for v in enc["embeddings"]],
        })

    section("STEP 1b — /encode (is_query=false)")
    doc = post(RETRIEVER_URL, "/encode", {
        "texts": [
            "custom binary operations with associativity testing",
            "geometry with constraints and ratios",
            "Polynomial interpolation with periodic value constraints",
        ],
        "is_query": False,
    })
    print(f"  ok={doc.get('ok')}, count={len(doc.get('embeddings', []))}")
    if doc.get("embeddings"):
        dim = len(doc["embeddings"][0])
        save("01b_encode_doc", {
            "ok": doc["ok"],
            "embeddings_shape": [len(doc["embeddings"]), dim],
            "embeddings_preview": [v[:8] for v in doc["embeddings"]],
        })

    section("STEP 1c — /rank (mode=embedding)")
    candidates = [
        {"problem_type": "custom binary operations with associativity testing", "utility": 0.5},
        {"problem_type": "geometry with constraints and ratios",                "utility": 0.3},
        {"problem_type": "Polynomial interpolation with periodic value constraints", "utility": 0.1},
        {"problem_type": "combinatorics with constraints",                      "utility": 0.8},
        {"problem_type": "algebra system of equations",                         "utility": 0.4},
    ]
    rank_emb = post(RETRIEVER_URL, "/rank", {
        "question": "How to solve a custom defined binary operation?",
        "candidates": candidates, "mode": "embedding", "top_k": 3,
    })
    print(f"  ok={rank_emb.get('ok')}, ranked_indices={rank_emb.get('ranked_indices')}")
    for idx in rank_emb.get("ranked_indices", []):
        print(f"    [{idx}] {candidates[idx]['problem_type']!r}")
    save("01c_rank_embedding", {"request_candidates": candidates, "response": rank_emb})

    section("STEP 1d — /rank (mode=hybrid, lambda=0.8)")
    rank_hyb = post(RETRIEVER_URL, "/rank", {
        "question": "How to solve a custom defined binary operation?",
        "candidates": candidates, "mode": "hybrid", "retrieve_lambda": 0.8, "top_k": 3,
    })
    print(f"  ok={rank_hyb.get('ok')}, ranked_indices={rank_hyb.get('ranked_indices')}")
    for idx in rank_hyb.get("ranked_indices", []):
        print(f"    [{idx}] {candidates[idx]['problem_type']!r}  utility={candidates[idx]['utility']}")
    save("01d_rank_hybrid", {"request_candidates": candidates, "response": rank_hyb})

    section("STEP 1e — /encode 错误场景（texts 为空）")
    err = post(RETRIEVER_URL, "/encode", {"texts": [], "is_query": True})
    print(f"  ok={err.get('ok')}, error={err.get('error')}")
    save("01e_encode_error", err)

    section("STEP 1f — /health 确认 idle 计时器已重置")
    rh2 = get(RETRIEVER_URL, "/health")
    print(f"  idle_remaining={rh2.get('idle_remaining')}s  (应接近 {rh2.get('idle_timeout')}s)")
    save("01f_retriever_health_after_calls", rh2)


# ─── STEP 2：添加 skills，触发警告区降级 ──────────────────────────────────────

def test_add_with_eviction() -> None:
    section("STEP 2 — 添加 Skills（触发主库满 → 警告区降级）")
    add_results = []

    for i, skill in enumerate(SKILLS_TO_ADD):
        result = post(MEMORY_URL, "/add", skill)
        zone = result.get("zone", "?")
        evicted = result.get("evicted_id")
        print(
            f"  [{i+1}] {skill['skill name']!r}  "
            f"ok={result.get('ok')}  id={result.get('id')}  "
            f"zone={zone}" + (f"  evicted_id={evicted}" if evicted else "")
        )
        add_results.append({"input_skill_name": skill["skill name"], "response": result})

    save("02_add_with_eviction", add_results)

    # 验证：前3条进主库，后2条进警告区
    zones = [r["response"].get("zone") for r in add_results]
    assert zones[:3] == ["main", "main", "main"], f"前3条应进主库，实际: {zones[:3]}"
    assert zones[3:] == ["warning", "warning"], f"后2条应进警告区，实际: {zones[3:]}"
    print("  [OK] 主库/警告区分配符合预期")

    snapshot = print_zone_summary("add后")
    save("02b_zone_snapshot_after_add", snapshot)

    # 测试：警告区再加1条（容量=2已满），应触发淘汰
    section("STEP 2c — 警告区满时再 add（应淘汰警告区最低 utility）")
    extra_skill = {
        "skill name": "Quadratic formula application",
        "problem type": "algebra quadratic equations",
        "key insight": "Apply the quadratic formula directly when factoring is not obvious.",
        "method": "1. Identify a,b,c. 2. Apply x=(-b±sqrt(b²-4ac))/(2a). 3. Simplify.",
        "skill_from": "success_rollout",
        "problem": "Solve 2x^2 - 3x - 2 = 0.",
        "reward": 0.6,
    }
    result_extra = post(MEMORY_URL, "/add", extra_skill)
    print(
        f"  extra skill → zone={result_extra.get('zone')}  "
        f"evicted_id={result_extra.get('evicted_id')}"
    )
    assert result_extra.get("zone") == "warning", "警告区满时新 skill 仍应进警告区"
    assert result_extra.get("evicted_id") is not None, "警告区满时应触发淘汰"
    print("  [OK] 警告区满时触发淘汰（符合预期）")
    save("02c_warn_eviction", result_extra)

    snapshot2 = print_zone_summary("warn eviction后")
    save("02d_zone_snapshot_after_eviction", snapshot2)


# ─── STEP 3：检索（主库+警告区合并） ─────────────────────────────────────────

def test_retrieve_combined() -> None:
    section("STEP 3 — /retrieve（主库+警告区合并检索）")
    queries = [
        "custom binary operation",
        "geometry window square panes",
        "polynomial interpolation periodic pattern",
        "counting arrangements forbidden adjacency",
        "algebra equations substitution symmetry",
    ]
    results = []
    for q in queries:
        r = post(MEMORY_URL, "/retrieve", {"question": q, "top_k": 5})
        print(f"  query: {q!r}")
        for sk in r.get("skills", []):
            print(f"    - {sk.get('skill name')!r}")
        results.append({"query": q, "response": r})
    save("03_retrieve_combined", results)

    # 验证：检索到的条数 = 主库(3) + 警告区(2) = 5
    total_retrieved = results[0]["response"].get("count", 0)
    print(f"  第一个 query 返回条数: {total_retrieved}（期望 ≤5）")


# ─── STEP 4：警告区 + is_success=False → 彻底删除 ────────────────────────────

def test_warn_remove_on_failure() -> None:
    section("STEP 4 — 警告区 skill 失败时彻底删除")

    # 查询当前警告区中的 skill
    ws_before = get_warn_status()
    warn_skills = ws_before.get("skills", [])
    if not warn_skills:
        print("  [SKIP] 警告区为空，跳过此测试")
        save("04_warn_remove_skip", {"reason": "warn zone empty"})
        return

    target = warn_skills[0]
    target_id = target.get("id")
    target_name = target.get("skill name")
    print(f"  选取警告区第1条: {target_name!r}  id={target_id!r}")
    print(f"  更新前警告区大小: {ws_before.get('warn_size')}")

    result = post(MEMORY_URL, "/update", {"skills": [
        {"id": target_id, "is_success": False, "reward": 0.8},
    ]})
    r = result.get("results", [{}])[0]
    print(
        f"  ok={r.get('ok')}  zone={r.get('zone')}  action={r.get('action')}  "
        f"utility: {r.get('utility_before')} -> {r.get('utility_after')}"
    )
    assert r.get("action") == "removed", f"警告区 + 失败 应 action=removed，实际: {r.get('action')}"
    print("  [OK] 警告区 skill 失败 → 彻底删除（符合预期）")
    save("04_warn_remove_on_failure", result)

    ws_after = get_warn_status()
    print(f"  更新后警告区大小: {ws_after.get('warn_size')}")
    assert ws_after.get("warn_size", 0) < ws_before.get("warn_size", 0), \
        "删除后警告区大小应减少"
    print("  [OK] 警告区大小已减少")
    save("04b_zone_snapshot_after_remove", {"before": ws_before, "after": ws_after})


# ─── STEP 5：警告区 + is_success=True + utility 不够 → 留在警告区 ─────────────

def test_warn_stay_on_low_utility() -> None:
    section("STEP 5 — 警告区 skill 成功但 utility 不足以晋升 → 留在警告区")

    ws = get_warn_status()
    warn_skills = ws.get("skills", [])
    if not warn_skills:
        print("  [SKIP] 警告区为空，跳过此测试")
        save("05_warn_stay_skip", {"reason": "warn zone empty"})
        return

    target = warn_skills[0]
    target_id = target.get("id")
    target_name = target.get("skill name")
    utility_before = target.get("utility", 0.0)
    print(f"  选取: {target_name!r}  id={target_id!r}  utility_before={utility_before}")

    # reward 低于 tau → effective_reward=0 → 不更新，应 action=no_change
    result = post(MEMORY_URL, "/update", {"skills": [
        {"id": target_id, "is_success": True, "reward": REWARD_BELOW_TAU},
    ]})
    r = result.get("results", [{}])[0]
    print(f"  action={r.get('action')}  utility: {r.get('utility_before')} -> {r.get('utility_after')}")
    assert r.get("action") in ("no_change", "stayed"), \
        f"effective_reward=0 时 action 应为 no_change/stayed，实际: {r.get('action')}"
    assert r.get("zone") == "warning", f"该 skill 应在 warning 区，实际: {r.get('zone')}"
    print("  [OK] reward<tau → 警告区 skill utility 未变（符合预期）")
    save("05_warn_stay_low_utility", result)

    ws_after = get_warn_status()
    still_in_warn = any(it.get("skill name") == target_name for it in ws_after.get("skills", []))
    assert still_in_warn, "skill 应仍在警告区"
    print("  [OK] skill 仍在警告区")
    save("05b_zone_snapshot", {"before_warn": ws, "after_warn": ws_after})


# ─── STEP 6：警告区 + is_success=True + utility 足够 → 晋升到主库 ─────────────

def test_warn_promote_on_high_utility() -> None:
    section("STEP 6 — 警告区 skill 成功且 utility 足以晋升 → swap 回主库")

    ws = get_warn_status()
    warn_skills = ws.get("skills", [])
    if not warn_skills:
        print("  [SKIP] 警告区为空，跳过此测试")
        save("06_warn_promote_skip", {"reason": "warn zone empty"})
        return

    # 挑 utility 最高的 warn skill，给高 reward，使其晋升
    target = max(warn_skills, key=lambda x: x.get("utility", 0.0))
    target_id = target.get("id")
    target_name = target.get("skill name")
    print(f"  选取警告区最高 utility: {target_name!r}  id={target_id!r}  utility={target.get('utility')}")

    st_before = get_main_status()
    print(f"  晋升前主库大小: {st_before.get('current_size')}")

    # 给足够高的 reward 使 new_utility > 主库最低 utility
    result = post(MEMORY_URL, "/update", {"skills": [
        {"id": target_id, "is_success": True, "reward": 1.0},
    ]})
    r = result.get("results", [{}])[0]
    print(
        f"  ok={r.get('ok')}  zone={r.get('zone')}  action={r.get('action')}\n"
        f"  utility: {r.get('utility_before')} -> {r.get('utility_after')}\n"
        f"  promoted_from_warn_id={r.get('promoted_from_warn_id')}\n"
        f"  demoted_to_warn_id={r.get('demoted_to_warn_id')}\n"
        f"  evicted_warn_id={r.get('evicted_warn_id')}"
    )
    save("06_warn_promote", result)

    if r.get("action") == "promoted":
        print("  [OK] skill 成功晋升到主库（符合预期）")
        ws_after = get_warn_status()
        st_after = get_main_status()
        print(f"  晋升后 主库:{st_after.get('current_size')} 警告区:{ws_after.get('warn_size')}")
        promoted_still_in_warn = any(
            it.get("skill name") == target_name for it in ws_after.get("skills", [])
        )
        assert not promoted_still_in_warn, "晋升后 skill 不应还在警告区"
        print("  [OK] 晋升后 skill 已移出警告区")
        save("06b_zone_snapshot_after_promote", {
            "main_status": st_after,
            "warn_status": ws_after,
        })
    else:
        print(f"  [INFO] action={r.get('action')}（可能当前 utility 未超过主库最低）")
        save("06b_zone_snapshot_no_promote", {"result": r, "warn_status": ws})


# ─── STEP 7：主库 skill 正常 EMA 更新 ─────────────────────────────────────────

def test_main_normal_update() -> None:
    section("STEP 7 — 主库 skill 正常 EMA 更新")

    # 查询主库中有哪些 skill（用 list_ids + get 查）
    ids_resp = post(MEMORY_URL, "/manage", {"action": "list_ids"})
    ids = ids_resp.get("ids", [])
    if not ids:
        print("  [SKIP] 主库为空")
        save("07_main_update_skip", {"reason": "main empty"})
        return

    # 取第一个 id 查详细信息
    first_item = post(MEMORY_URL, "/manage", {"action": "get", "id": ids[0]})
    skill_id = ids[0]
    skill_name = first_item.get("skill", {}).get("skill name", "")
    if not skill_name:
        print("  [SKIP] 无法获取主库 skill name")
        return

    print(f"  选取主库 skill: {skill_name!r}  id={skill_id!r}")

    result = post(MEMORY_URL, "/update", {"skills": [
        {"id": skill_id, "is_success": True, "reward": 0.8},
    ]})
    r = result.get("results", [{}])[0]
    print(
        f"  ok={r.get('ok')}  zone={r.get('zone')}  action={r.get('action')}  "
        f"utility: {r.get('utility_before')} -> {r.get('utility_after')}"
    )
    assert r.get("zone") == "main", f"主库 skill 应在 main 区，实际: {r.get('zone')}"
    assert r.get("action") in ("updated", "no_change"), \
        f"主库 skill 应为 updated/no_change，实际: {r.get('action')}"
    print(f"  [OK] 主库 skill action={r.get('action')}（符合预期）")
    save("07_main_normal_update", result)


# ─── STEP 8：检索后对比（warn 区内容变化） ────────────────────────────────────

def test_retrieve_after_zone_changes() -> None:
    section("STEP 8 — 警告区变化后再次检索")
    queries = [
        "custom binary operation",
        "geometry window square panes",
        "polynomial interpolation periodic pattern",
    ]
    results = []
    for q in queries:
        r = post(MEMORY_URL, "/retrieve", {"question": q, "top_k": 5})
        print(f"  query: {q!r}")
        for sk in r.get("skills", []):
            print(f"    - {sk.get('skill name')!r}")
        results.append({"query": q, "response": r})
    save("08_retrieve_after_zone_changes", results)


# ─── STEP 9：/manage warn_status 完整快照 ─────────────────────────────────────

def test_final_status() -> None:
    section("STEP 9 — 最终状态快照")
    main_st = get_main_status()
    warn_st = get_warn_status()
    print(f"  主库: {main_st.get('current_size')}/{main_st.get('max_capacity')}  "
          f"is_full={main_st.get('is_full')}")
    print(f"  警告区: {warn_st.get('warn_size')}/{warn_st.get('warn_capacity')}")
    warn_names = [it.get("skill name") for it in warn_st.get("skills", [])]
    print(f"  警告区内容: {warn_names}")
    save("09a_final_main_status", main_st)
    save("09b_final_warn_status", warn_st)


# ─── STEP 10：retriever_server 空闲状态 ──────────────────────────────────────

def check_retriever_idle() -> None:
    section("STEP 10 — retriever_server 空闲计时器状态")
    rh = get(RETRIEVER_URL, "/health")
    print(f"  model_loaded={rh.get('model_loaded')}  idle_remaining={rh.get('idle_remaining')}s")
    save("10_retriever_health_final", rh)


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n[test_server] memory_server:    {MEMORY_URL}")
    print(f"[test_server] retriever_server:  {RETRIEVER_URL}")
    print(f"[test_server] 结果输出:           {OUTPUT_DIR}")
    print(f"\n  注意：建议以 --max-capacity 3 --warn-capacity 2 启动 memory_server")
    print(f"  这样添加5条 skill 后才能触发警告区降级和淘汰逻辑\n")

    wait_ready(RETRIEVER_URL, "retriever_server")
    wait_ready(MEMORY_URL,    "memory_server")

    check_health()
    test_retriever_directly()
    test_add_with_eviction()
    test_retrieve_combined()
    test_warn_remove_on_failure()
    test_warn_stay_on_low_utility()
    test_warn_promote_on_high_utility()
    test_main_normal_update()
    test_retrieve_after_zone_changes()
    test_final_status()
    check_retriever_idle()

    print(f"\n{'='*62}")
    print(f"  全部测试完成，结果已保存到 {OUTPUT_DIR}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
