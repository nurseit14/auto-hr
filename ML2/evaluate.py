"""Reproduce frozen-model evaluation without training or selecting parameters."""

import argparse
import json
from pathlib import Path

from ML2.ranker import LinearRanker
from ML2.train import ROOT, dataset_fingerprint, evaluate_methods, evaluate_hr_methods, load_dataset, matrix_for


def evaluate_saved(model_path, dataset, split="test", direction="cv_to_jobs"):
    if direction not in {"cv_to_jobs", "job_to_candidates"}:
        raise ValueError("Unknown ranking direction")
    model = LinearRanker.load(model_path)
    if model.metadata.get("dataset_fingerprints") != dataset_fingerprint(dataset):
        raise ValueError("Dataset has changed since training; use the frozen dataset")
    splits, _ = load_dataset(dataset)
    evaluator = evaluate_hr_methods if direction == "job_to_candidates" else evaluate_methods
    return evaluator(splits[split], matrix_for(splits[split]), model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверить сохранённый реранкер без переобучения")
    parser.add_argument("--model", type=Path, default=ROOT / "backend/models/ranker.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/annotations/ml2_training_v1")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--direction", choices=["cv_to_jobs", "job_to_candidates"], default="cv_to_jobs")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_saved(args.model, args.dataset, args.split, args.direction), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Ошибка: {error}")
