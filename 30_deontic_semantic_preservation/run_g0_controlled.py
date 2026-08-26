from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


LABELS = ("obl", "ent", "pro", "per", "pow", "dis", "none")
# Entitlement is deliberately excluded here. LexDeMod is actor-specific: a
# tenant obligation can be the landlord's entitlement, so blindly replacing
# the tenant's modal does not create an entitlement->obligation contrast for
# the annotated actor. The three classes below permit a gold-span-local edit.
HEADLINE = {"obl", "pro", "per"}
TRANSFORMS = {
    "obl": {
        "shall": ("may", "OBLIGATION_TO_PERMISSION"),
        "must": ("may", "OBLIGATION_TO_PERMISSION"),
        "is required to": ("may", "OBLIGATION_TO_PERMISSION"),
        "are required to": ("may", "OBLIGATION_TO_PERMISSION"),
        "has to": ("may", "OBLIGATION_TO_PERMISSION"),
        "have to": ("may", "OBLIGATION_TO_PERMISSION"),
    },
    "per": {
        "may": ("must", "PERMISSION_TO_OBLIGATION"),
        "is permitted to": ("must", "PERMISSION_TO_OBLIGATION"),
        "are permitted to": ("must", "PERMISSION_TO_OBLIGATION"),
        "is allowed to": ("must", "PERMISSION_TO_OBLIGATION"),
        "are allowed to": ("must", "PERMISSION_TO_OBLIGATION"),
    },
    "pro": {
        "shall not": ("may", "PROHIBITION_LOSS"),
        "must not": ("may", "PROHIBITION_LOSS"),
        "may not": ("may", "PROHIBITION_LOSS"),
        "will not": ("may", "PROHIBITION_LOSS"),
        "cannot": ("may", "PROHIBITION_LOSS"),
    },
}


def _gold_span_edit(text: str, spans: dict, label: str):
    """Edit only a LexDeMod gold trigger span, never a coincidental modal."""
    body = re.sub(r"^\[[^\]]+\]\s*", "", text.strip())
    tokens = body.split()
    for start, end in spans.get(label, []):
        if not (0 <= start < end <= len(tokens)):
            continue
        trigger = " ".join(tokens[start:end]).casefold()
        transform = TRANSFORMS[label].get(trigger)
        if transform is None:
            continue
        replacement, change_type = transform
        edited = tokens[:start] + [replacement] + tokens[end:]
        return body, " ".join(edited), change_type, [start, end], trigger
    return None


def build_pairs(path: Path, per_class: int, seed: int) -> list[dict]:
    pools = {key: [] for key in TRANSFORMS}
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            vector = ast.literal_eval(row["label"])
            active = {LABELS[index] for index, value in enumerate(vector) if value}
            headline = active & HEADLINE
            # Retain rows with exactly one active deontic class. This avoids a
            # second gold operator confounding the intended minimal contrast.
            if len(headline) != 1 or active - HEADLINE:
                continue
            label = next(iter(headline))
            spans = ast.literal_eval(row["span"])
            edit = _gold_span_edit(row["text"], spans, label)
            if edit:
                original, perturbed, change_type, gold_span, gold_trigger = edit
                key = (original, perturbed, change_type)
                if key not in seen:
                    seen.add(key)
                    pools[label].append(
                        {
                            "id": row.get("id"),
                            "cid": row.get("cid"),
                            "gold_label": label,
                            "change_type": change_type,
                            "gold_span": gold_span,
                            "gold_trigger": gold_trigger,
                            "original": original,
                            "perturbed": perturbed,
                        }
                    )
    rng = random.Random(seed)
    selected = []
    for label, pool in pools.items():
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    return selected


def lexical_metrics(rows: list[dict]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_text = [row["original"] for row in rows] + [row["perturbed"] for row in rows]
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(all_text)
    n = len(rows)
    for index, row in enumerate(rows):
        a = set(re.findall(r"\w+", row["original"].casefold()))
        b = set(re.findall(r"\w+", row["perturbed"].casefold()))
        row["token_jaccard"] = len(a & b) / max(len(a | b), 1)
        row["sequence_similarity"] = SequenceMatcher(None, row["original"], row["perturbed"]).ratio()
        row["tfidf_cosine"] = float(matrix[index].multiply(matrix[n + index]).sum())


def embed(model_path: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype="auto", device_map="auto")
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors)


def natural_pairs(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((str(row["original"]), str(row["simplified"])))
    return pairs


def describe(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "rate_ge_0_90": float((x >= 0.90).mean()),
        "rate_ge_0_95": float((x >= 0.95).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexdemod", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--natural-pairs", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    rows = build_pairs(args.lexdemod, args.per_class, args.seed)
    if not rows:
        raise RuntimeError("no controlled perturbations could be constructed")
    lexical_metrics(rows)
    originals = [row["original"] for row in rows]
    perturbed = [row["perturbed"] for row in rows]
    vectors = embed(args.embedding_model, originals + perturbed)
    n = len(rows)
    for index, row in enumerate(rows):
        row["embedding_cosine"] = float(np.dot(vectors[index], vectors[n + index]))

    natural = natural_pairs(args.natural_pairs)
    natural_similarity = []
    if natural:
        natural_vectors = embed(args.embedding_model, [x for pair in natural for x in pair])
        for index in range(len(natural)):
            natural_similarity.append(float(np.dot(natural_vectors[2 * index], natural_vectors[2 * index + 1])))

    by_type = {}
    for change_type in sorted({row["change_type"] for row in rows}):
        subset = [row for row in rows if row["change_type"] == change_type]
        by_type[change_type] = {
            "n": len(subset),
            "embedding_cosine": describe([row["embedding_cosine"] for row in subset]),
        }
    result = {
        "n_perturbations": len(rows),
        "class_counts": {key: sum(row["gold_label"] == key for row in rows) for key in TRANSFORMS},
        "token_jaccard": describe([row["token_jaccard"] for row in rows]),
        "sequence_similarity": describe([row["sequence_similarity"] for row in rows]),
        "tfidf_cosine": describe([row["tfidf_cosine"] for row in rows]),
        "embedding_cosine": describe([row["embedding_cosine"] for row in rows]),
        "natural_simplification_embedding_cosine": describe(natural_similarity) if natural_similarity else None,
        "by_change_type": by_type,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.joinpath("g0_controlled_records.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    args.out_dir.joinpath("g0_controlled_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
