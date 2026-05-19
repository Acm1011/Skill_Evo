from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from baselines.ReasoningBankMath.embedding_backend import cosine_similarity, embed_texts
from baselines.ReasoningBankMath.io_utils import read_jsonl
from baselines.ReasoningBankMath.trajectory_utils import (
    format_math_trajectory,
    normalize_trajectory_row,
)

from .memory_parser import memory_item_key, parse_teacher_output
from .text_utils import normalize_memory_text, normalize_space, short_hash, utc_now_iso


def build_embedding_text(
    query: str,
    topic: Any,
    status: str,
    raw_rule: str,
    items: Iterable[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    q = normalize_space(query)
    if q:
        parts.append(f"Question: {q}")
    t = normalize_space(topic)
    if t:
        parts.append(f"Topic: {t}")
    s = normalize_space(status)
    if s:
        parts.append(f"Status: {s}")
    rr = normalize_space(raw_rule)
    if rr:
        parts.append(f"Raw Rule: {rr}")
    for item in items:
        title = normalize_space(item.get("title"))
        desc = normalize_space(item.get("description"))
        content = normalize_space(item.get("content"))
        memory_type = normalize_space(item.get("memory_type"))
        line = " | ".join(x for x in (memory_type, title, desc, content) if x)
        if line:
            parts.append(line)
    return "\n".join(parts)


def make_memory_record(
    group: Dict[str, Any],
    parsed: Dict[str, Any],
    *,
    raw_teacher_output: str,
) -> Dict[str, Any]:
    query = str(group.get("problem") or "").strip()
    topic = group.get("topic")
    topic_key = group.get("topic_key") or "unknown"
    status = str(group.get("status") or "unknown")
    memory_type = str(group.get("memory_type") or "unknown")
    memory_items = list(parsed.get("memory_items") or [])
    raw_rule = str(parsed.get("raw_rule") or "").strip()
    source_indices = [row.get("idx", row.get("line_idx")) for row in group.get("rows", [])]
    created_at = utc_now_iso()
    memory_id = f"mem_{short_hash(f'{topic_key}|{memory_type}|{query}|{status}|{raw_rule}|{raw_teacher_output}')}"
    provenance = []
    for row in group.get("rows", []):
        provenance.append(
            {
                "source_idx": row.get("idx", row.get("line_idx")),
                "line_idx": row.get("line_idx"),
                "status": "success" if row.get("is_correct") is True else "failure",
                "query": row.get("problem"),
            }
        )
    representative = group["rows"][0]
    return {
        "memory_id": memory_id,
        "source_idx": source_indices[0] if source_indices else None,
        "query": query,
        "topic": topic,
        "topic_key": topic_key,
        "status": status,
        "memory_type": memory_type,
        "trajectory": format_math_trajectory(
            query,
            str(representative.get("student_response") or ""),
            representative.get("ground_truth"),
        ),
        "raw_rule": raw_rule,
        "memory_items": memory_items,
        "embedding_text": build_embedding_text(query, topic, status, raw_rule, memory_items),
        "raw_teacher_output": raw_teacher_output,
        "provenance": provenance,
        "duplicate_count": 0,
        "created_at": created_at,
        "updated_at": created_at,
    }


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for line_idx, row in enumerate(rows):
        norm = normalize_trajectory_row(row, line_idx)
        if norm is not None:
            out.append(norm)
    return out


def item_signature_set(record: Dict[str, Any]) -> set[str]:
    return {memory_item_key(item) for item in record.get("memory_items", []) if isinstance(item, dict)}


def record_signature(record: Dict[str, Any]) -> str:
    keys = sorted(item_signature_set(record))
    return "##".join(keys)


def memory_type_set(record: Dict[str, Any]) -> set[str]:
    out = set()
    for item in record.get("memory_items", []):
        if isinstance(item, dict):
            value = normalize_memory_text(item.get("memory_type"))
            if value:
                out.add(value)
    if not out:
        top = normalize_memory_text(record.get("memory_type"))
        if top:
            out.add(top)
    return out


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
                old_types = memory_type_set(merged[best_idx])
                new_types = memory_type_set(rec)
                if old_types and new_types and old_types == new_types:
                    old_sig = item_signature_set(merged[best_idx])
                    new_sig = item_signature_set(rec)
                    if old_sig & new_sig:
                        matched_id = str(merged[best_idx].get("memory_id"))
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

