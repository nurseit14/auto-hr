"""Evaluate an explicitly synthetic, independently authored HR ranking exercise.

This is not a validation set for model selection and not a study of real HR
performance. Keep its fixture frozen before evaluating a candidate model.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from ML2.features import extract_features
from ML2.metrics import evaluate
from ML2.ranker import LinearRanker


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = ROOT / "ML2/tests/fixtures/hr_ranking_cases.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixture.get("schema_version") != "autohr-synthetic-hr-ranking-v1":
        raise ValueError("Unknown synthetic benchmark schema")
    cases = fixture.get("cases", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Expected nonempty distinct vacancy cases")
    for case in cases:
        candidates = case["candidates"]
        ids = {candidate["id"] for candidate in candidates}
        if len(ids) != len(candidates) or not candidates:
            raise ValueError("Candidate IDs must be distinct within a vacancy")
        for candidate in candidates:
            if not candidate["text"].strip() or candidate["relevance"] not in (0, 1, 2, 3):
                raise ValueError("Synthetic candidates need source text and valid evidence label")
        for preference in case.get("preferences", []):
            if preference["higher"] not in ids or preference["lower"] not in ids:
                raise ValueError("Preference refers to an unknown candidate")
    return fixture


def evaluate_hr_benchmark(model, fixture_path=DEFAULT_FIXTURE, model_path=None):
    fixture = load_cases(fixture_path)
    rows, predictions, by_case = [], [], []
    total_preferences = strict_correct = tied = 0
    case_preference_credits = []
    for case in fixture["cases"]:
        matrix = [extract_features(candidate["text"], case["vacancy"])[0]
                  for candidate in case["candidates"]]
        scores = [float(value) for value in model.predict(matrix)]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("Model produced a nonfinite synthetic benchmark score")
        by_id = {candidate["id"]: score for candidate, score in zip(case["candidates"], scores)}
        for candidate, score in zip(case["candidates"], scores):
            # The shared metric groups by cv_id; here a query is one vacancy.
            rows.append({"cv_id": case["id"], "relevance": candidate["relevance"]})
            predictions.append(score)
        credits = []
        for preference in case.get("preferences", []):
            higher, lower = by_id[preference["higher"]], by_id[preference["lower"]]
            equal = math.isclose(higher, lower, abs_tol=1e-12, rel_tol=0.)
            credit = .5 if equal else float(higher > lower)
            strict_correct += int(not equal and higher > lower)
            tied += int(equal)
            total_preferences += 1
            credits.append(credit)
        if credits:
            case_preference_credits.append(sum(credits) / len(credits))
        by_case.append({"case_id": case["id"],
                        "preference_credit": sum(credits) / len(credits) if credits else None,
                        "scores": by_id})
    metrics = evaluate(rows, predictions, k=5)
    details = metrics.pop("by_query")
    report = {
        "schema_version": "autohr-synthetic-hr-evaluation-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_sha256": file_sha256(fixture_path),
        "fixture_schema": fixture["schema_version"],
        "model_sha256": file_sha256(model_path) if model_path else None,
        "model_version": model.to_dict()["model_version"],
        "feature_version": model.to_dict()["feature_version"],
        "score_policy_version": model.to_dict().get("score_policy_version"),
        "vacancy_queries": len(fixture["cases"]),
        "candidate_pairs": len(rows),
        "metrics": metrics,
        "predeclared_preferences": {
            "count": total_preferences,
            "strict_correct": strict_correct,
            "ties": tied,
            "incorrect": total_preferences - strict_correct - tied,
            "macro_credit": sum(case_preference_credits) / len(case_preference_credits)
                            if case_preference_credits else None,
        },
        "limitations": [
            "Synthetic engineering expectations, not independent professional HR labels or real hiring outcomes",
            "Only ten authored IT vacancy cases; not an estimate of population or language-specific accuracy",
            "Screening-evidence labels describe available text, not actual human ability or candidate rejection",
            "Evaluation-only fixture; do not use these scores to select model hyperparameters",
            "Any later bug fixes informed by these examples turn them into regression cases, requiring a new independent evaluation",
        ],
    }
    return report, {"metric_queries": details, "case_scores": by_case}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate synthetic candidate-to-vacancy ranking")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", type=Path, default=ROOT / "backend/models/ranker.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", action="store_true", help="Include per-case scores for error analysis")
    args = parser.parse_args(argv)
    report, details = evaluate_hr_benchmark(LinearRanker.load(args.model), args.fixture, args.model)
    if args.details:
        report["details"] = details
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("Output already exists; preserve evaluations and choose a new output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Benchmark error: {error}")
