"""Train a local pilot reranker on the frozen reviewed pair dataset."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ML2.features import FEATURE_NAMES, extract_features
from ML2.metrics import evaluate
from ML2.prepare_dataset import SCHEMA_VERSION, sha256, write_json, write_jsonl
from ML2.ranker import LinearRanker, MODEL_VERSION


ROOT = Path(__file__).resolve().parent.parent
FROZEN_FILES = ["pairs_all.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl",
                "manifest.json", "candidate_groups.json", "split_assignment.json"]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_fingerprint(path):
    return {name: hashlib.sha256((Path(path) / name).read_bytes()).hexdigest() for name in FROZEN_FILES}


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown dataset schema; use ML2.prepare_dataset first")
    source = read_jsonl(path / "pairs_all.jsonl")
    by_id = {r["pair_id"]: r for r in source}
    if len(by_id) != len(source):
        raise ValueError("Duplicate source pair IDs")
    groups = json.loads((path / "candidate_groups.json").read_text(encoding="utf-8"))
    lookup = {cid: gid for gid, ids in groups.items() for cid in ids}
    if sum(map(len, groups.values())) != len(lookup):
        raise ValueError("A CV belongs to multiple candidate groups")
    for row in source:
        if lookup.get(row["cv_id"]) != row["candidate_group_id"]:
            raise ValueError("Candidate group mapping is inconsistent")
        if sha256(row["cv_text"]) != row["cv_text_sha256"] or sha256(row["vacancy_display"]) != row["vacancy_text_sha256"]:
            raise ValueError("Frozen source text hash mismatch")
    splits, seen_pairs, seen_candidates = {}, set(), set()
    for split in ("train", "validation", "test"):
        frozen = read_jsonl(path / f"{split}.jsonl")
        rows = []
        for identity in frozen:
            pid = identity["pair_id"]
            if pid not in by_id or pid in seen_pairs:
                raise ValueError("Unknown pair or pair overlap between splits")
            row = by_id[pid]
            if any(row.get(key) != value for key, value in identity.items()):
                raise ValueError("Split identities/targets disagree with source pairs")
            if (row["overall_score"] not in (1, 2, 3) or row["relevance"] != row["overall_score"] - 1
                    or not row["judged"] or not row["cv_text"].strip()):
                raise ValueError("Unknown or invalid target in a ranking split")
            seen_pairs.add(pid)
            rows.append(row)
        if not rows:
            raise ValueError(f"Empty {split} split")
        candidates = {r["candidate_group_id"] for r in rows}
        if candidates & seen_candidates:
            raise ValueError("Candidate leakage between splits")
        seen_candidates.update(candidates)
        if set(Counter(r["cv_id"] for r in rows).values()) != {5}:
            raise ValueError("Each CV query must retain its five annotated vacancies")
        splits[split] = rows
    eligible = {r["pair_id"] for r in source if r["judged"] and r["cv_text"].strip()}
    if eligible != seen_pairs:
        raise ValueError("Frozen splits do not cover exactly the eligible pairs")
    return splits, manifest


def matrix_for(rows):
    return [extract_features(row["cv_text"], row)[0] for row in rows]


def method_scores(rows, matrix, model=None):
    results = {"skill_overlap": [v[0] for v in matrix]}
    if all(r.get("selection", {}).get("selection_rank") is not None for r in rows):
        results["original_shortlist_order"] = [-float(r["selection"]["selection_rank"]) for r in rows]
    if model is not None:
        results["trained_ml2"] = model.predict(matrix).tolist()
    return results


def evaluate_methods(rows, matrix, model):
    return {name: evaluate(rows, scores) for name, scores in method_scores(rows, matrix, model).items()}


def train_dataset(dataset, output, l2_grid=(.01, .1, 1., 10.)):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new experiment version")
    splits, dataset_manifest = load_dataset(dataset)
    fingerprints = dataset_fingerprint(dataset)
    # Training/validation features are built before any test labels are evaluated.
    train_matrix, validation_matrix = matrix_for(splits["train"]), matrix_for(splits["validation"])
    candidates = []
    for l2 in l2_grid:
        model = LinearRanker.fit(splits["train"], train_matrix, l2=l2)
        metrics = evaluate(splits["validation"], model.predict(validation_matrix))
        if metrics["ndcg_at_5_informative"] is None:
            raise ValueError("Validation needs at least one informative CV query")
        candidates.append((model, metrics))
    model, validation = max(candidates, key=lambda item: (
        item[1]["ndcg_at_5_informative"], item[1]["pairwise_accuracy"], item[0].metadata["l2"]))
    model.metadata.update({"trained_at": datetime.now(timezone.utc).isoformat(),
                           "train_pairs": len(splits["train"]),
                           "train_candidate_groups": len({r["candidate_group_id"] for r in splits["train"]}),
                           "dataset_fingerprints": fingerprints,
                           "dataset_source_sha256": dataset_manifest["source_sha256"],
                           "selection_metric": "validation ndcg_at_5_informative; pairwise accuracy; stronger L2",
                           "label_semantics": dataset_manifest["label_mapping"],
                           "feature_policy": "Fixed professional concepts only; no identity, contacts, demographics, annotation explanations or scores as inputs",
                           "experimental": True})
    # The chosen model is frozen before its only test evaluation in this run.
    test_matrix = matrix_for(splits["test"])
    report = {"model_version": MODEL_VERSION, "experimental": True,
              "dataset_fingerprints": fingerprints,
              "evaluation_scope": dataset_manifest["evaluation_scope"],
              "metric_policy": "Linear relevance gains 0/1/2; expected DCG for tied scores; all-weak queries excluded from NDCG; informative queries have differing labels",
              "validation_search": [{"l2": m.metadata["l2"], "metrics": metrics}
                                    for m, metrics in candidates],
              "selected_l2": model.metadata["l2"],
              "train": evaluate_methods(splits["train"], train_matrix, model),
              "validation": evaluate_methods(splits["validation"], validation_matrix, model),
              "test": evaluate_methods(splits["test"], test_matrix, model),
              "limitations": ["Small pilot; test contains only six CV queries in the current dataset",
                              "Some test vacancies also occur in training; this is not an unseen-vacancy test",
                              "Original shortlist is BM25/skills, not an executed ML1 embedding baseline",
                              "Dictionary features do not reproduce every requirement judgment or generate new span annotations",
                              "Employment duration uses explicitly stated totals, not inferred job intervals",
                              "Only ten good-fit pairs in the complete eligible dataset; mixed human/assistant review"]}
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.json")
    write_json(output / "evaluation.json", report)
    predictions = []
    scores = method_scores(splits["test"], test_matrix, model)
    for index, row in enumerate(splits["test"]):
        predictions.append({k: row[k] for k in ("pair_id", "cv_id", "vacancy_id", "overall_score", "relevance")}
                           | {name: values[index] for name, values in scores.items()})
    write_jsonl(output / "test_predictions.jsonl", predictions)
    lines = ["ML2 — первый локальный обученный реранкер", "",
             f"Обучение: {len(splits['train'])} пар. Настройка: {len(splits['validation'])}. Тест: {len(splits['test'])}.",
             "Модель: линейное попарное ранжирование по профессиональным признакам.",
             "Балл модели задаёт порядок вакансий; это не вероятность найма и не оценка 0–3.", "",
             "Тест: NDCG@5 на резюме с разными оценками (ближе к 1 — лучше):"]
    for name, metrics in report["test"].items():
        lines.append(f"  {name}: {metrics['ndcg_at_5_informative']:.4f}")
    lines += ["", "Тестовый набор небольшой: вывод о готовности к реальным решениям по нему не делается.",
              "Обучаемая модель сортирует вакансии. Предразметка требований остаётся отдельным модулем."]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Обучить первый локальный реранкер ML2")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/annotations/ml2_training_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/annotations/ml2_reranker_v1")
    args = parser.parse_args(argv)
    report = train_dataset(args.dataset, args.output_dir)
    print("Модель сохранена:", args.output_dir / "model.json")
    print(json.dumps({name: {k: v for k, v in metrics.items() if k != "by_query"}
                      for name, metrics in report["test"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Ошибка: {error}")
