"""Reproduce frozen-model evaluation without training or selecting parameters."""

import argparse
import json
from pathlib import Path

from ML2.ranker import LinearRanker
from ML2.train import ROOT, dataset_fingerprint, evaluate_methods, load_dataset, matrix_for


def evaluate_saved(model_path, dataset, split="test"):
    model = LinearRanker.load(model_path)
    if model.metadata.get("dataset_fingerprints") != dataset_fingerprint(dataset):
        raise ValueError("Dataset has changed since training; use the frozen dataset")
    splits, _ = load_dataset(dataset)
    return evaluate_methods(splits[split], matrix_for(splits[split]), model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверить сохранённый реранкер без переобучения")
    parser.add_argument("--model", type=Path, default=ROOT / "data/annotations/ml2_reranker_v1/model.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/annotations/ml2_training_v1")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_saved(args.model, args.dataset, args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Ошибка: {error}")
