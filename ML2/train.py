"""Train a local pilot reranker on the frozen reviewed pair dataset."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from ML2.features import FEATURE_NAMES, extract_features
from ML2.metrics import evaluate
from ML2.prepare_dataset import SCHEMA_VERSION, sha256, write_json, write_jsonl
from ML2.ranker import LinearRanker, MODEL_VERSION, SCORE_POLICY_VERSION, preference_pairs


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


def evaluate_hr_methods(rows, matrix, model):
    # A job's original position within a CV's shortlist is not a comparable
    # candidate score across different CVs and is deliberately not a HR baseline.
    return {name: evaluate(rows, scores, group_by="vacancy_id")
            for name, scores in method_scores(rows, matrix, model).items()
            if name != "original_shortlist_order"}


def candidate_folds(rows, count=3, seed=42):
    """Deterministic candidate-group folds; labels never affect assignment."""
    groups = sorted({row["candidate_group_id"] for row in rows},
                    key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    count = min(count, len(groups))
    if count < 2:
        raise ValueError("At least two candidate groups are required for model selection")
    assignment = {group: index % count for index, group in enumerate(groups)}
    return [(np.asarray([i for i, row in enumerate(rows) if assignment[row["candidate_group_id"]] != fold]),
             np.asarray([i for i, row in enumerate(rows) if assignment[row["candidate_group_id"]] == fold]))
            for fold in range(count)]


def aggregate_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "by_query"}


def selection_evaluation(rows, scores):
    return {"cv_to_jobs": aggregate_metrics(evaluate(rows, scores)),
            "job_to_candidates": aggregate_metrics(evaluate(rows, scores, group_by="vacancy_id"))}


def select_configuration(rows, matrix, l2_grid, hr_grid=(0., .5), grade_grid=(False, True)):
    """Predeclared, bounded selection inside train, without the historical test.

    Each fold fits scales and coefficients only on its training candidate groups.
    Primary selection is macro NDCG across the two informative query directions;
    ties use pairwise accuracy, stronger L2, then the simpler objective.
    """
    matrix = np.asarray(matrix, dtype=float)
    folds = candidate_folds(rows)
    candidates = []
    for l2 in l2_grid:
        for hr_weight in hr_grid:
            for grade_gap in grade_grid:
                fold_metrics = []
                for train, heldout in folds:
                    fit_rows, check_rows = [rows[i] for i in train], [rows[i] for i in heldout]
                    model = LinearRanker.fit(fit_rows, matrix[train], l2=l2,
                                             hr_weight=hr_weight, grade_gap=grade_gap)
                    fold_metrics.append(selection_evaluation(check_rows, model.predict(matrix[heldout])))
                direction_scores, direction_pairs = {}, {}
                for direction in ("cv_to_jobs", "job_to_candidates"):
                    ndcg = [item[direction]["ndcg_at_5_informative"] for item in fold_metrics
                            if item[direction]["ndcg_at_5_informative"] is not None]
                    accuracy = [item[direction]["pairwise_accuracy"] for item in fold_metrics
                                if item[direction]["pairwise_accuracy"] is not None]
                    direction_scores[direction] = float(np.mean(ndcg)) if ndcg else None
                    direction_pairs[direction] = float(np.mean(accuracy)) if accuracy else None
                informative = [value for value in direction_scores.values() if value is not None]
                pairwise = [value for value in direction_pairs.values() if value is not None]
                if not informative:
                    raise ValueError("Candidate-group cross-validation has no informative queries")
                candidates.append({"l2": l2, "hr_weight": hr_weight, "grade_gap": grade_gap,
                                   "cv_macro_ndcg": float(np.mean(informative)),
                                   "cv_macro_pairwise_accuracy": float(np.mean(pairwise)) if pairwise else 0.,
                                   "direction_ndcg": direction_scores,
                                   "direction_pairwise_accuracy": direction_pairs,
                                   "fold_metrics": fold_metrics})
    selected = max(candidates, key=lambda item: (
        item["cv_macro_ndcg"], item["cv_macro_pairwise_accuracy"], item["l2"],
        -item["hr_weight"], not item["grade_gap"]))
    return selected, candidates


def train_dataset(dataset, output, l2_grid=(.01, .1, 1., 10.), locked_configuration=None,
                  development_provenance=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new experiment version")
    splits, dataset_manifest = load_dataset(dataset)
    fingerprints = dataset_fingerprint(dataset)
    # Training/validation features are built before any test labels are evaluated.
    train_matrix, validation_matrix = matrix_for(splits["train"]), matrix_for(splits["validation"])
    if locked_configuration is None:
        selected, candidates = select_configuration(splits["train"], train_matrix, l2_grid)
        selection_description = "Three candidate-group folds within train; mean informative NDCG across both directions, pairwise accuracy, stronger L2, simpler objective"
    else:
        # A heldout-informed correctness fix must not reopen model search.
        # Recheck only the previously locked configuration on the corrected
        # source; report that check separately from the original selection.
        if set(locked_configuration) != {"l2", "hr_weight", "grade_gap"}:
            raise ValueError("A locked configuration needs exactly l2, hr_weight and grade_gap")
        l2, hr_weight, grade_gap = (locked_configuration[key] for key in ("l2", "hr_weight", "grade_gap"))
        if not isinstance(l2, (int, float)) or not np.isfinite(l2) or l2 <= 0 or not 0 <= hr_weight <= 1 or not isinstance(grade_gap, bool):
            raise ValueError("Invalid locked training configuration")
        selected, recheck = select_configuration(splits["train"], train_matrix, (float(l2),),
                                                 hr_grid=(float(hr_weight),), grade_grid=(grade_gap,))
        candidates = []
        selection_description = "Previously locked train-group-CV configuration, refit after a heldout-informed extraction correctness fix; no hyperparameter reselection"
    model = LinearRanker.fit(splits["train"], train_matrix, l2=selected["l2"],
                             hr_weight=selected["hr_weight"], grade_gap=selected["grade_gap"])
    validation = selection_evaluation(splits["validation"], model.predict(validation_matrix))
    model.metadata.update({"trained_at": datetime.now(timezone.utc).isoformat(),
                           "train_pairs": len(splits["train"]),
                           "train_candidate_groups": len({r["candidate_group_id"] for r in splits["train"]}),
                           "training_direction_pair_counts": {
                               "cv_to_jobs": len(preference_pairs(splits["train"], np.asarray(train_matrix), hr_weight=0.)[0]),
                               "job_to_candidates": len(preference_pairs(splits["train"], np.asarray(train_matrix), hr_weight=1.)[0])},
                           "dataset_fingerprints": fingerprints,
                           "dataset_source_sha256": dataset_manifest["source_sha256"],
                           "selection_metric": selection_description,
                           "model_selection": {"folds": 3, "seed": 42,
                                               "selected": {key: value for key, value in selected.items() if key != "fold_metrics"},
                                               "configuration_count": len(candidates),
                                               "fixed_configuration_recheck": locked_configuration is not None,
                                               "validation_used_for_selection": False,
                                               "test_used_for_selection": False},
                           "validation_prediction_policy": SCORE_POLICY_VERSION,
                           "label_semantics": dataset_manifest["label_mapping"],
                           "feature_policy": "Role-aware technical evidence gates level and explicitly scoped experience; mixed totals are not allocated; no identity, contacts, demographics, annotation explanations or scores as inputs",
                           "experimental": True})
    if development_provenance:
        model.metadata["development_provenance"] = development_provenance
    # Persist the choice and weights before constructing any test features.
    # The historical test is descriptive after earlier project reuse, not a
    # pristine prospective holdout, but must still never select this model.
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.json")
    write_json(output / "selection_lock.json", {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": hashlib.sha256((output / "model.json").read_bytes()).hexdigest(),
        "selected": {key: value for key, value in selected.items() if key != "fold_metrics"},
        "test_used_for_selection": False,
        "independent_hr_benchmark_used_for_selection": False,
        "development_provenance": development_provenance or {}})
    # The chosen model is frozen before its only test evaluation in this run.
    test_matrix = matrix_for(splits["test"])
    report = {"model_version": MODEL_VERSION, "experimental": True,
              "score_policy_version": SCORE_POLICY_VERSION,
              "score_policy": "Positive linear evidence is multiplied by the explicitly established relevant-experience minimum fulfillment; unknown duration and negative linear bases are unchanged. This is a declared rule, not a learned coefficient.",
              "dataset_fingerprints": fingerprints,
              "evaluation_scope": dataset_manifest["evaluation_scope"],
              "metric_policy": "Linear relevance gains 0/1/2; expected DCG for tied scores; all-weak queries excluded from NDCG; informative queries have differing labels",
              "cross_validation_search": candidates,
              "fixed_configuration_cross_validation_recheck": selected if locked_configuration is not None else None,
              "development_provenance": development_provenance or {},
              "validation_search": [{"l2": selected["l2"], "hr_weight": selected["hr_weight"],
                                     "grade_gap": selected["grade_gap"], "metrics": validation}],
              "selected_l2": model.metadata["l2"],
              "train": evaluate_methods(splits["train"], train_matrix, model),
              "validation": evaluate_methods(splits["validation"], validation_matrix, model),
              "test": evaluate_methods(splits["test"], test_matrix, model),
              "hr_ranking": {"train": evaluate_hr_methods(splits["train"], train_matrix, model),
                             "validation": evaluate_hr_methods(splits["validation"], validation_matrix, model),
                             "test": evaluate_hr_methods(splits["test"], test_matrix, model)},
              "limitations": ["Small pilot; test contains only six CV queries in the current dataset",
                              "Some test vacancies also occur in training; this is not an unseen-vacancy test",
                              "Original shortlist is BM25/skills, not an executed ML1 embedding baseline",
                              "Dictionary features do not reproduce every requirement judgment or generate new span annotations",
                              "Employment duration uses explicitly role-scoped statements or totals in one supported profile, not inferred job intervals",
                              "Zero input evidence anchors score at zero; positive scores remain uncalibrated relative evidence, not hiring probabilities",
                              "The explicit relevant-experience shortfall rule is engineering policy, not a learned coefficient; validation and reported metrics include this rule",
                              "Only ten good-fit pairs in the complete eligible dataset; mixed human/assistant review",
                              "Three train candidate-group folds selected the configuration before validation/test reporting; fixed validation is descriptive and did not select hyperparameters",
                              "HR-direction metrics are sparse; show informative and multi-candidate query counts, not singleton-inflated overall NDCG",
                              "The historical pilot test was already reused in earlier project development; it is not a pristine prospective holdout"]}
    if development_provenance:
        report["limitations"].append("This extraction correctness fix was informed by the first synthetic HR holdout; that first holdout is now a regression set, not independent quality evidence for the revised source")
    write_json(output / "evaluation.json", report)
    predictions = []
    scores = method_scores(splits["test"], test_matrix, model)
    for index, row in enumerate(splits["test"]):
        predictions.append({k: row[k] for k in ("pair_id", "cv_id", "vacancy_id", "overall_score", "relevance")}
                           | {name: values[index] for name, values in scores.items()})
    write_jsonl(output / "test_predictions.jsonl", predictions)
    lines = ["ML2 — локальный обученный реранкер с учётом профиля вакансии", "",
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/annotations/ml2_reranker_v3")
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
