from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .embedding_backend import cosine_similarity, embed_texts
from .io_utils import read_jsonl
from .memory_parser import parse_case_output, plan_signature
from .text_utils import json_dumps_compact, normalize_memory_text, normalize_space, short_hash, utc_now_iso
from .trajectory_utils import format_math_trajectory, normalize_trajectory_row


def build_embedding_text(record: Dict[str, Any]) -> str:
    parts: List[str] = [
        f"Question: {normalize_space(record.get('query'))}",
        f"Topic: {normalize_space(record.get('topic'))}",
        f"Status: {normalize_space(record.get('status'))}",
        f"Takeaway: {normalize_space(record.get('takeaway'))}",
    ]
    for step in record.get("plan_steps", []):
        desc = normalize_space(step.get("description"))
        if desc:
            parts.append(desc)
    return "\n".join(x for x in parts if x and not x.endswith(": "))


def make_memory_record(
    traj: Dict[str, Any],
    parsed_case: Dict[str, Any],
    *,
    raw_teacher_output: str,
) -> Dict[str, Any]:
    query = str(traj.get("problem") or "").strip()
    topic = traj.get("topic")
    topic_key = traj.get("topic_key") or "unknown"
    status = "success" if traj.get("is_correct") is True else "failure"
    reward = 1 if status == "success" else 0
    case_label = "positive" if reward == 1 else "negative"
    source_idx = traj.get("idx", traj.get("line_idx"))
    created_at = utc_now_iso()
    plan_steps = list(parsed_case.get("plan_steps") or [])
    plan_json = {"plan": [{"id": i + 1, "description": x["description"]} for i, x in enumerate(plan_steps)]}
    memory_id = f"mem_{short_hash(f'{topic_key}|{query}|{status}|{json_dumps_compact(plan_json)}')}"
    record = {
        "memory_id": memory_id,
        "source_idx": source_idx,
        "query": query,
        "topic": topic,
        "topic_key": topic_key,
        "status": status,
        "reward": reward,
        "case_label": case_label,
        "case_summary": parsed_case.get("case_summary") or f"{status} math case",
        "trajectory": format_math_trajectory(
            query,
            str(traj.get("student_response") or ""),
            traj.get("ground_truth"),
        ),
        "plan": plan_json,
        "plan_steps": plan_json["plan"],
        "takeaway": parsed_case.get("takeaway") or "",
        "tags": list(parsed_case.get("tags") or []),
        "created_at": created_at,
        "updated_at": created_at,
        "raw_teacher_output": raw_teacher_output,
        "provenance": [
            {
                "source_idx": source_idx,
                "line_idx": traj.get("line_idx"),
                "status": status,
                "query": query,
            }
        ],
        "duplicate_count": 0,
    }
    record["embedding_text"] = build_embedding_text(record)
    return record


def export_case_pool_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case": record.get("query"),
        "plan": json_dumps_compact(record.get("plan") or {"plan": []}),
        "case_label": record.get("case_label"),
    }


def export_dummy_memory_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": record.get("query"),
        "plan": json_dumps_compact(record.get("plan") or {"plan": []}),
        "reward": int(record.get("reward") or 0),
    }


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for line_idx, row in enumerate(rows):
        norm = normalize_trajectory_row(row, line_idx)
        if norm is not None:
            out.append(norm)
    return out


def parse_teacher_output(raw: str, *, fallback_status: str) -> Dict[str, Any]:
    return parse_case_output(raw, fallback_status=fallback_status)


def record_signature(record: Dict[str, Any]) -> str:
    query = normalize_memory_text(record.get("query"))
    status = normalize_space(record.get("status")).lower()
    plan_sig = plan_signature(list(record.get("plan_steps") or []))
    return f"{query}##{status}##{plan_sig}"


def append_provenance(existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    existing.setdefault("provenance", [])
    for prov in incoming.get("provenance", []):
        if prov not in existing["provenance"]:
            existing["provenance"].append(prov)
    existing["duplicate_count"] = int(existing.get("duplicate_count") or 0) + 1
    existing["updated_at"] = utc_now_iso()


def dedupe_records(
    existing_records: List[Dict[str, Any]],
    new_records: List[Dict[str, Any]],
    *,
    similarity_threshold: float = 0.98,
    embed_backend: str = "hash",
    embed_base_url: str = "",
    embed_api_key: str = "",
    embed_model: str = "",
    embed_timeout: float = 600.0,
    embed_dim: int = 256,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    merged = [dict(r) for r in existing_records]
    signature_to_id: Dict[str, str] = {}
    id_to_index: Dict[str, int] = {}
    for i, rec in enumerate(merged):
        signature_to_id[record_signature(rec)] = str(rec.get("memory_id"))
        id_to_index[str(rec.get("memory_id"))] = i

    if similarity_threshold > 0 and merged:
        existing_embs = embed_texts(
            [str(rec.get("embedding_text") or "") for rec in merged],
            backend=embed_backend,
            base_url=embed_base_url,
            api_key=embed_api_key,
            model=embed_model,
            timeout=embed_timeout,
            dim=embed_dim,
        )
    else:
        existing_embs = []

    duplicate_map: Dict[str, str] = {}
    for rec in new_records:
        sig = record_signature(rec)
        if sig in signature_to_id:
            target_id = signature_to_id[sig]
            append_provenance(merged[id_to_index[target_id]], rec)
            duplicate_map[str(rec.get("memory_id"))] = target_id
            continue

        matched_id: Optional[str] = None
        if similarity_threshold > 0 and existing_embs:
            new_emb = embed_texts(
                [str(rec.get("embedding_text") or "")],
                backend=embed_backend,
                base_url=embed_base_url,
                api_key=embed_api_key,
                model=embed_model,
                timeout=embed_timeout,
                dim=embed_dim,
            )[0]
            best_score = -1.0
            best_idx = -1
            for idx, old_emb in enumerate(existing_embs):
                score = cosine_similarity(new_emb, old_emb)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score >= similarity_threshold:
                old = merged[best_idx]
                if (
                    normalize_space(old.get("topic_key")) == normalize_space(rec.get("topic_key"))
                    and normalize_space(old.get("status")) == normalize_space(rec.get("status"))
                ):
                    matched_id = str(old.get("memory_id"))
        if matched_id:
            append_provenance(merged[id_to_index[matched_id]], rec)
            duplicate_map[str(rec.get("memory_id"))] = matched_id
            continue

        merged.append(rec)
        signature_to_id[sig] = str(rec.get("memory_id"))
        id_to_index[str(rec.get("memory_id"))] = len(merged) - 1
        if similarity_threshold > 0:
            existing_embs.append(
                embed_texts(
                    [str(rec.get("embedding_text") or "")],
                    backend=embed_backend,
                    base_url=embed_base_url,
                    api_key=embed_api_key,
                    model=embed_model,
                    timeout=embed_timeout,
                    dim=embed_dim,
                )[0]
            )

    return merged, duplicate_map
