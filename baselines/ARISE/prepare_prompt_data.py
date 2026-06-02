from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from baselines.SkillRL.prepare_rl_data import retrieve_bucket
from skill_src.prepare_test_data import _apply_first_user_content

from .skill_bank import AriseSkillBank, format_skill_prompt, load_skill_use_template


def run_prepare_prompt_data(args: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("prepare-prompt-data 需要 pandas、pyarrow: pip install pandas pyarrow") from e

    from skill_src.skill_manager.skill_controller import SkillController

    bank = AriseSkillBank.from_path(args.library_json, include_reservoir=args.include_reservoir)
    candidates = bank.build_candidates()
    if not candidates:
        raise SystemExit(f"prepare-prompt-data: no usable skills found in {args.library_json}")
    template_text = load_skill_use_template()

    input_path = Path(args.input_jsonl)
    if not input_path.is_file():
        raise SystemExit(f"prepare-prompt-data: input not found: {input_path}")

    kept_rows: List[Dict[str, Any]] = []
    total = 0
    kept = 0
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            extra = row.get("extra_info")
            if not isinstance(extra, dict):
                continue
            question = extra.get("problem")
            if not isinstance(question, str) or not question.strip():
                continue
            question = question.strip()

            try:
                retrieved_skills = retrieve_bucket(
                    question=question,
                    candidates=candidates,
                    top_k=args.top_k,
                    retriever_url=args.retriever_url,
                    mode=args.mode,
                    retrieve_lambda=args.retrieve_lambda,
                )
            except Exception as e:
                if args.fail_on_retrieve_error:
                    raise SystemExit(f"prepare-prompt-data: retrieve failed at line {line_no}: {e}") from e
                continue

            skill_block = format_skill_prompt(retrieved_skills)
            if not skill_block.strip():
                continue
            new_content = template_text.format(skill=skill_block, question=question)

            rec = copy.deepcopy(row)
            prompt = rec.get("prompt")
            if isinstance(prompt, list):
                prompt = [m for m in prompt if not (isinstance(m, dict) and m.get("role") == "system")]
                rec["prompt"] = prompt
            if not _apply_first_user_content(prompt, new_content):
                continue

            ex = rec.get("extra_info")
            if not isinstance(ex, dict):
                ex = {}
                rec["extra_info"] = ex
            ex["retrieval_mode"] = args.mode
            ex["retriever_url"] = args.retriever_url
            ex["top_k"] = args.top_k
            ex["skill_id"] = [str(s.get("skill_id") or "") for s in retrieved_skills]
            ex["retrieved_skill_ids"] = list(ex["skill_id"])
            ex["retrieved_skill_count"] = len(retrieved_skills)
            ex["skill_candidates_count"] = len(candidates)
            kept_rows.append(rec)
            kept += 1

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in kept_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows_for_parquet = copy.deepcopy(kept_rows)
    for rec in rows_for_parquet:
        ex = rec.get("extra_info")
        if isinstance(ex, dict):
            SkillController._sanitize_extra_info_for_parquet(ex)
        SkillController._parquet_flatten_prompt_and_extra_info(rec)
        SkillController._coerce_remaining_nested_to_json_strings(rec)
    out_parquet = Path(args.output_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_for_parquet).to_parquet(out_parquet, index=False)

    print(f"[arise-prepare-prompt-data] wrote {out_jsonl} ({kept}/{total})")
    print(f"[arise-prepare-prompt-data] wrote {out_parquet} ({kept}/{total})")
    return 0


def build_prepare_prompt_data_parser(sub: Any) -> None:
    p = sub.add_parser("prepare-prompt-data", help="Inject retrieved ARISE skills into temp/greedy jsonl")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--library-json", required=True, help="ARISE skill library checkpoint json")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8766")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--mode", default="embedding", choices=["embedding", "hybrid"])
    p.add_argument("--retrieve-lambda", type=float, default=0.5)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--include-reservoir", action="store_true", help="Also retrieve from reservoir entries")
    p.add_argument("--fail-on-retrieve-error", action="store_true")
    p.set_defaults(_run=run_prepare_prompt_data)
