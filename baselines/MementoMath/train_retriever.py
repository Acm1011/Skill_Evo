from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class MemoryRetrieverClassifier(nn.Module):
    def __init__(self, sentence_bert: AutoModel):
        super().__init__()
        hidden = sentence_bert.config.hidden_size
        self.sentence_bert = sentence_bert
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, ids1, mask1, ids2, mask2):
        o1 = self.sentence_bert(ids1, attention_mask=mask1).last_hidden_state[:, 0]
        o2 = self.sentence_bert(ids2, attention_mask=mask2).last_hidden_state[:, 0]
        return self.classifier(torch.cat([o1, o2], dim=1))


def _parse_plan(plan_field: Union[str, dict, list, None]) -> Optional[Union[dict, list]]:
    if plan_field is None:
        return None
    if isinstance(plan_field, (dict, list)):
        return plan_field
    if isinstance(plan_field, str):
        s = plan_field.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return {"plan": [{"description": s}]}
    return None


def _pretty_plan(plan_obj: Union[dict, list]) -> str:
    try:
        steps = []
        if isinstance(plan_obj, dict) and "plan" in plan_obj and isinstance(plan_obj["plan"], list):
            for item in plan_obj["plan"]:
                if isinstance(item, dict):
                    sid = item.get("id")
                    desc = item.get("description") or item.get("desc") or item.get("step") or str(item)
                    steps.append(f"{sid}. {desc}" if sid is not None else f"- {desc}")
                else:
                    steps.append(f"- {str(item)}")
        elif isinstance(plan_obj, list):
            for i, item in enumerate(plan_obj, 1):
                if isinstance(item, dict):
                    desc = item.get("description") or item.get("desc") or item.get("step") or str(item)
                    steps.append(f"{i}. {desc}")
                else:
                    steps.append(f"{i}. {str(item)}")
        else:
            return json.dumps(plan_obj, ensure_ascii=False)
        return "\n".join(steps) if steps else json.dumps(plan_obj, ensure_ascii=False)
    except Exception:
        return json.dumps(plan_obj, ensure_ascii=False)


class PairJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        use_plan: bool = True,
        plan_style: str = "pretty",
        section_tokens: Tuple[str, str] = ("[CASE]", "[PLAN]"),
    ):
        self.samples: List[Dict[str, Any]] = []
        st_case, st_plan = section_tokens
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                query = obj.get("query")
                case = obj.get("case")
                plan = obj.get("plan", None)
                label = obj.get("truth_label")
                if query is None or case is None or label is None:
                    raise ValueError(f"[line {ln}] missing query/case/truth_label")

                y = int(label) if isinstance(label, (int, str)) else (1 if label else 0)
                y = 1 if y == 1 else 0

                icl_parts = [st_case, str(case)]
                if use_plan and plan is not None:
                    icl_parts.append(st_plan)
                    if plan_style == "pretty":
                        pobj = _parse_plan(plan)
                        icl_parts.append(_pretty_plan(pobj) if pobj is not None else str(plan))
                    else:
                        icl_parts.append(str(plan))
                icl_text = "\n".join(icl_parts).strip()
                self.samples.append({"natural": str(query), "icl": icl_text, "label": y})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def labels(self) -> List[int]:
        return [x["label"] for x in self.samples]


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int = 256

    def __call__(self, batch: List[Dict[str, Any]]):
        icl_texts = [b["icl"] for b in batch]
        nat_texts = [b["natural"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        t1 = self.tokenizer(icl_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        t2 = self.tokenizer(nat_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return t1["input_ids"], t1["attention_mask"], t2["input_ids"], t2["attention_mask"], labels


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_probs, all_labels = [], []
    for ids1, mask1, ids2, mask2, labels in loader:
        ids1, mask1 = ids1.to(device), mask1.to(device)
        ids2, mask2 = ids2.to(device), mask2.to(device)
        labels = labels.to(device)
        probs = torch.softmax(model(ids1, mask1, ids2, mask2), dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


def stratified_split_indices(labels: List[int], val_ratio: float, seed: int, stratify: bool = True):
    n = len(labels)
    idx_all = list(range(n))
    if not stratify:
        random.Random(seed).shuffle(idx_all)
        k = max(1, int(n * val_ratio))
        return idx_all[k:], idx_all[:k]

    rng = random.Random(seed)
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k_pos = max(1, int(len(pos) * val_ratio)) if len(pos) > 0 else 0
    k_neg = max(1, int(len(neg) * val_ratio)) if len(neg) > 0 else 0
    val_idx = pos[:k_pos] + neg[:k_neg]
    train_idx = pos[k_pos:] + neg[k_neg:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    if len(train_idx) == 0 and len(val_idx) > 1:
        train_idx, val_idx = val_idx[1:], val_idx[:1]
    return train_idx, val_idx


def count_pos_in_subset(dataset: PairJsonlDataset, indices: List[int]) -> int:
    return sum(dataset.samples[i]["label"] for i in indices)


def run_train_retriever(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tok_src = args.pretrained_local or args.pretrained_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    backbone = AutoModel.from_pretrained(tok_src)
    model = MemoryRetrieverClassifier(backbone).to(device)
    collate = Collator(tokenizer, max_len=args.max_len)
    if args.valid:
        train_ds = PairJsonlDataset(
            args.train, use_plan=args.use_plan, plan_style=args.plan_style, section_tokens=tuple(args.section_tokens)
        )
        valid_ds = PairJsonlDataset(
            args.valid, use_plan=args.use_plan, plan_style=args.plan_style, section_tokens=tuple(args.section_tokens)
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
        pos = sum(x["label"] for x in train_ds.samples)
        neg = len(train_ds) - pos
    else:
        full_ds = PairJsonlDataset(
            args.train, use_plan=args.use_plan, plan_style=args.plan_style, section_tokens=tuple(args.section_tokens)
        )
        tr_idx, va_idx = stratified_split_indices(
            full_ds.labels(), val_ratio=args.val_ratio, seed=args.seed, stratify=not args.no_stratify
        )
        train_ds = Subset(full_ds, tr_idx)
        valid_ds = Subset(full_ds, va_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)
        pos = count_pos_in_subset(full_ds, tr_idx)
        neg = len(tr_idx) - pos

    no_decay = ["bias", "LayerNorm.weight", "layer_norm", "ln"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    if args.class_weight_pos is not None:
        weight = torch.tensor([1.0, float(args.class_weight_pos)], device=device)
    else:
        weight = None
        if pos > 0 and neg > 0:
            weight = torch.tensor([1.0, float(max(1.0, neg / max(1, pos)))], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    global_step, best_metric = 0, -1.0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            ids1, mask1, ids2, mask2, labels = batch
            ids1, mask1 = ids1.to(device), mask1.to(device)
            ids2, mask2 = ids2.to(device), mask2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(ids1, mask1, ids2, mask2)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                metrics = evaluate(model, valid_loader, device)
                score = metrics["auc"] if not math.isnan(metrics["auc"]) else metrics["f1"]
                print(
                    f"[step {global_step}] val_acc={metrics['acc']:.4f} "
                    f"val_f1={metrics['f1']:.4f} val_auc={metrics['auc']:.4f}"
                )
                if args.save_best and score > best_metric:
                    best_metric = score
                    torch.save(model.state_dict(), best_path)
                    print(f"  -> best updated: {best_path}")

        metrics = evaluate(model, valid_loader, device)
        score = metrics["auc"] if not math.isnan(metrics["auc"]) else metrics["f1"]
        print(
            f"[epoch {epoch}] loss={epoch_loss / max(1, len(train_loader)):.4f} "
            f"val_acc={metrics['acc']:.4f} val_f1={metrics['f1']:.4f} val_auc={metrics['auc']:.4f}"
        )
        if args.save_best and score > best_metric:
            best_metric = score
            torch.save(model.state_dict(), best_path)
            print(f"  -> best updated: {best_path}")

    final_path = os.path.join(args.output_dir, "last.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training done. Final: {final_path}")
    if args.save_best and os.path.exists(best_path):
        print(f"Best checkpoint: {best_path}")
    return 0


def build_train_retriever_parser(sub: Any) -> None:
    p = sub.add_parser("train-retriever", help="Train Memento-style parametric retriever locally")
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--valid", type=str, default=None)
    p.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    p.add_argument("--pretrained-name", dest="pretrained_name", type=str, default="princeton-nlp/sup-simcse-roberta-base")
    p.add_argument("--pretrained-local", dest="pretrained_local", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.1)
    p.add_argument("--no-stratify", dest="no_stratify", action="store_true")
    p.add_argument("--use-plan", dest="use_plan", action="store_true")
    p.add_argument("--plan-style", dest="plan_style", type=str, default="pretty", choices=["pretty", "raw"])
    p.add_argument("--section-tokens", dest="section_tokens", nargs=2, default=["[CASE]", "[PLAN]"])
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", dest="warmup_ratio", type=float, default=0.06)
    p.add_argument("--max-len", dest="max_len", type=int, default=256)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--eval-every", dest="eval_every", type=int, default=500)
    p.add_argument("--save-best", dest="save_best", action="store_true")
    p.add_argument("--class-weight-pos", dest="class_weight_pos", type=float, default=None)
    p.set_defaults(_run=run_train_retriever)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=False)
    build_train_retriever_parser(sub)
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    if getattr(args, "_run", None) is None:
        args = parser.parse_args(["train-retriever"] + (argv or []))
    return int(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main())
