from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .train_retriever import MemoryRetrieverClassifier, _parse_plan, _pretty_plan


class CaseRetriever:
    def __init__(
        self,
        model_path: str,
        model_name: str = "princeton-nlp/sup-simcse-roberta-base",
        device: str | None = None,
        score_batch_size: int = 32,
        max_length: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_batch_size = max(1, int(score_batch_size))
        self.max_length = max(8, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        backbone = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.model = MemoryRetrieverClassifier(backbone).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.inference_mode()
    def _score_batch(self, natural: List[str], icl: List[str]) -> torch.Tensor:
        t1 = self.tokenizer(
            icl,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        t2 = self.tokenizer(
            natural,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids1, mask1 = t1["input_ids"].to(self.device), t1["attention_mask"].to(self.device)
        ids2, mask2 = t2["input_ids"].to(self.device), t2["attention_mask"].to(self.device)
        logits = self.model(ids1, mask1, ids2, mask2)
        return torch.softmax(logits, dim=1)[:, 1]

    def retrieve(self, natural_prompt: str, icl_pool: List[str], metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prob_chunks: List[torch.Tensor] = []
        for start in range(0, len(icl_pool), self.score_batch_size):
            sub_pool = icl_pool[start : start + self.score_batch_size]
            sub_nat = [natural_prompt] * len(sub_pool)
            prob_chunks.append(self._score_batch(sub_nat, sub_pool).detach().cpu())
        probs = torch.cat(prob_chunks, dim=0) if prob_chunks else torch.empty(0)
        results = []
        for i, (prompt, score, meta) in enumerate(zip(icl_pool, probs, metadata)):
            results.append(
                {
                    "prompt": prompt,
                    "score": float(score),
                    "index": i,
                    "case_label": meta.get("case_label", "unknown"),
                    "case": meta.get("case", ""),
                    "plan": meta.get("plan", None),
                }
            )
        return results


def build_icl_text(case: str, plan) -> str:
    parts = ["[CASE]", str(case)]
    if plan is not None:
        pobj = _parse_plan(plan)
        parts += ["[PLAN]", _pretty_plan(pobj) if pobj is not None else str(plan)]
    return "\n".join(parts).strip()


def load_pool(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    pool = []
    metadata = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case = obj.get("case")
            if case is None:
                raise ValueError("Each line in pool jsonl must contain 'case' field")
            plan = obj.get("plan", None)
            pool.append(build_icl_text(case, plan))
            metadata.append({"case": case, "plan": plan, "case_label": obj.get("case_label", "unknown")})
    if not pool:
        raise ValueError("Pool is empty")
    return pool, metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--pool-jsonl", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model-name", default="princeton-nlp/sup-simcse-roberta-base")
    ap.add_argument("--score-batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    retriever = CaseRetriever(
        model_path=args.model_path,
        model_name=args.model_name,
        score_batch_size=args.score_batch_size,
        max_length=args.max_length,
    )
    icl_pool, metadata = load_pool(args.pool_jsonl)
    ranked = retriever.retrieve(args.query, icl_pool, metadata)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    topk = ranked[: args.topk] if 0 < args.topk < len(ranked) else ranked
    for i, item in enumerate(topk, 1):
        print(f"[{i}] score={item['score']:.4f} idx={item['index']} label={item['case_label']}")
        print(f"{item['prompt']}\n" + "-" * 60)


if __name__ == "__main__":
    main()
