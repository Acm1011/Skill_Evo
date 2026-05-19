from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .embedding_backend import cosine_similarity, embed_texts
from .io_utils import read_jsonl
from .memory_parser import memory_item_key, parse_memory_items
from .text_utils import normalize_memory_text, normalize_space, short_hash, utc_now_iso
from .trajectory_utils import format_math_trajectory, normalize_trajectory_row


def build_embedding_text(query: str, topic: Any, items: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    q = normalize_space(query)
    if q:
        parts.append(f"Question: {q}")
    t = normalize_space(topic)
    if t:
        parts.append(f"Topic: {t}")
    for item in items:
        title = normalize_space(item.get("title"))
        desc = normalize_space(item.get("description"))
        content = normalize_space(item.get("content"))
        line = " | ".join(x for x in (title, desc, content) if x)
        if line:
            parts.append(line)
    return "\n".join(parts)


def make_memory_record(
    traj: Dict[str, Any],
    memory_items: List[Dict[str, Any]],
    *,
    raw_teacher_output: str,
) -> Dict[str, Any]:
    query = str(traj.get("problem") or "").strip()
    topic = traj.get("topic")
    topic_key = traj.get("topic_key") or "unknown"
    status = "success" if traj.get("is_correct") is True else "failure"
    source_idx = traj.get("idx", traj.get("line_idx"))
    created_at = utc_now_iso()
    memory_id = f"mem_{short_hash(f'{topic_key}|{query}|{status}|{raw_teacher_output}')}"
    return {
        "memory_id": memory_id,
        "source_idx": source_idx,
        "query": query,
        "topic": topic,
        "topic_key": topic_key,
        "status": status,
        "trajectory": format_math_trajectory(
            query,
            str(traj.get("student_response") or ""),
            traj.get("ground_truth"),
        ),
        "memory_items": memory_items,
        "created_at": created_at,
        "updated_at": created_at,
        "embedding_text": build_embedding_text(query, topic, memory_items),
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


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for line_idx, row in enumerate(rows):
        norm = normalize_trajectory_row(row, line_idx)
        if norm is not None:
            out.append(norm)
    return out


def parse_teacher_output(raw: str) -> List[Dict[str, str]]:
    items = parse_memory_items(raw)
    if not items:
        raise RuntimeError("teacher output did not contain any memory items")
    return items


def item_signature_set(record: Dict[str, Any]) -> set[str]:
    return {memory_item_key(item) for item in record.get("memory_items", []) if isinstance(item, dict)}


def record_signature(record: Dict[str, Any]) -> str:
    keys = sorted(item_signature_set(record))
    return "##".join(keys)


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

