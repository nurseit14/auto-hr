"""Small nonnegative linear ranker with a pairwise logistic training objective."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from ML2.features import FEATURE_NAMES, FEATURE_VERSION, extract_features, feature_fingerprint


MODEL_VERSION = "ml2-pairwise-linear-v1"


def preference_pairs(rows, matrix):
    queries = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("relevance") is None or row.get("overall_score") == 0:
            raise ValueError("Unknown labels cannot train the ranker")
        queries[row["cv_id"]].append(index)
    differences, weights = [], []
    for indices in queries.values():
        pairs = [(a, b) for a in indices for b in indices
                 if rows[a]["relevance"] > rows[b]["relevance"]]
        for a, b in pairs:
            differences.append(matrix[a] - matrix[b])
            weights.append(1. / len(pairs))
    if not differences:
        raise ValueError("Need at least one CV with different relevance labels")
    return np.asarray(differences), np.asarray(weights) / sum(weights)


class LinearRanker:
    def __init__(self, mean, scale, weights, metadata=None):
        self.mean, self.scale, self.weights = [np.asarray(v, dtype=float) for v in (mean, scale, weights)]
        if any(v.shape != (len(FEATURE_NAMES),) or not np.isfinite(v).all()
               for v in (self.mean, self.scale, self.weights)):
            raise ValueError("Invalid model dimensions or nonfinite parameters")
        if np.any(self.scale <= 0) or np.any(self.weights < 0):
            raise ValueError("Model scales must be positive and weights nonnegative")
        self.metadata = metadata or {}

    @classmethod
    def fit(cls, rows, matrix, l2=.1, iterations=3000):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
            raise ValueError("Invalid training feature matrix")
        if l2 <= 0:
            raise ValueError("Positive L2 regularization is required")
        mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
        scale[scale < 1e-8] = 1.
        diffs, sample_weight = preference_pairs(rows, (matrix - mean) / scale)
        weights = np.zeros(matrix.shape[1])
        lipschitz = .25 * np.linalg.norm(diffs * np.sqrt(sample_weight[:, None]), ord=2) ** 2 + l2
        for iteration in range(iterations):
            margin = diffs @ weights
            sigmoid_negative = np.exp(-np.logaddexp(0., margin))
            gradient = -(diffs.T @ (sample_weight * sigmoid_negative)) + l2 * weights
            updated = np.maximum(0., weights - gradient / lipschitz)
            if np.max(np.abs(updated - weights)) < 1e-9:
                weights = updated
                break
            weights = updated
        loss = float(np.dot(sample_weight, np.logaddexp(0., -(diffs @ weights)))
                     + .5 * l2 * (weights @ weights))
        return cls(mean, scale, weights, {"l2": l2, "iterations": iteration + 1,
                                         "preference_pairs": len(diffs), "training_objective": loss})

    def predict(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
            raise ValueError("Invalid prediction feature matrix")
        return ((matrix - self.mean) / self.scale) @ self.weights

    def to_dict(self):
        return {"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
                "feature_fingerprint": feature_fingerprint(),
                "feature_names": FEATURE_NAMES, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "weights": self.weights.tolist(),
                "score_semantics": "Relative ranking score, not probability or a 0–3 annotation grade",
                "metadata": self.metadata}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("model_version") != MODEL_VERSION or data.get("feature_version") != FEATURE_VERSION
                or data.get("feature_names") != FEATURE_NAMES
                or data.get("feature_fingerprint") != feature_fingerprint()):
            raise ValueError("Incompatible model/feature schema")
        return cls(data["mean"], data["scale"], data["weights"], data.get("metadata"))

    def rerank(self, cv_text, vacancies, top_k=5):
        """Accept source vacancies or ML1.match() results; preserve original IDs/scores."""
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not cv_text.strip():
            raise ValueError("Cannot rank for an empty CV")
        if not vacancies:
            return []
        ids = [str(v.get("vacancy_id", v.get("id", ""))) for v in vacancies]
        if any(not vid for vid in ids) or len(ids) != len(set(ids)):
            raise ValueError("Vacancy IDs must be present and unique")
        extracted = [extract_features(cv_text, v) for v in vacancies]
        scores = self.predict([v[0] for v in extracted])
        output = []
        for original, vid, (values, facts), score in zip(vacancies, ids, extracted, scores):
            contributions = (np.asarray(values) - self.mean) / self.scale * self.weights
            output.append({**original, "vacancy_id": vid, "ml2_score": float(score),
                           "ml2_model": MODEL_VERSION, "ml2_evidence": facts,
                           "ml2_contributions": dict(zip(FEATURE_NAMES, contributions.tolist()))})
        output.sort(key=lambda r: (-r["ml2_score"], r["vacancy_id"]))
        return output[:top_k]
