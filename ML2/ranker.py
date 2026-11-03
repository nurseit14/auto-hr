"""Small nonnegative linear ranker with a pairwise logistic training objective."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from ML2.features import FEATURE_NAMES, FEATURE_VERSION, extract_features, feature_fingerprint


MODEL_VERSION = "ml2-pairwise-linear-v2"
SCORE_POLICY_VERSION = "relevant-experience-shortfall-v1"


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
        training_mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
        scale[scale < 1e-8] = 1.
        # Pairwise differences are translation invariant. Anchor zero at absent
        # professional evidence rather than the average of a small mixed pilot.
        # This changes the score origin, not learned pairwise preferences.
        mean = np.zeros(matrix.shape[1])
        diffs, sample_weight = preference_pairs(rows, matrix / scale)
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
                                         "preference_pairs": len(diffs), "training_objective": loss,
                                         "score_origin": "Zero feature vector is zero evidence score; pairwise learning is translation invariant",
                                         "training_feature_mean": training_mean.tolist(),
                                         "score_policy_version": SCORE_POLICY_VERSION,
                                         "policy_semantics": "Declared rule, not a learned weight: an explicit relevant-experience shortfall scales only a positive linear score by the fulfilled minimum fraction; unknowns and negative bases are unchanged"})

    def _validated_matrix(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
            raise ValueError("Invalid prediction feature matrix")
        return matrix

    @staticmethod
    def _experience_policy(matrix, raw_scores):
        role = matrix[:, FEATURE_NAMES.index("role_compatibility")]
        experience = matrix[:, FEATURE_NAMES.index("experience_compatibility")]
        applies = (role > 0) & (experience < 0)
        factors = np.ones(len(matrix))
        factors[applies] = np.clip(1. + experience[applies] / role[applies], 0., 1.)
        adjustments = -np.maximum(raw_scores, 0.) * (1. - factors)
        return factors, adjustments, applies

    def predict(self, matrix):
        matrix = self._validated_matrix(matrix)
        raw = ((matrix - self.mean) / self.scale) @ self.weights
        _, adjustments, _ = self._experience_policy(matrix, raw)
        return raw + adjustments

    def explain(self, matrix):
        """Exact additive accounting, using the same validated input as predict."""
        matrix = self._validated_matrix(matrix)
        contributions = ((matrix - self.mean) / self.scale) * self.weights
        raw_scores = contributions.sum(axis=1)
        factors, adjustments, applies = self._experience_policy(matrix, raw_scores)
        return [{"base_score": 0., "raw_score": float(score),
                 "final_score": float(score + adjustments[row]),
                 "adjustments": [{"policy": SCORE_POLICY_VERSION,
                                  "factor": float(factors[row]),
                                  "contribution": float(adjustments[row])}] if applies[row] else [],
                 "contributions": [{"feature": name, "value": float(matrix[row, i]),
                                    "mean": float(self.mean[i]), "scale": float(self.scale[i]),
                                    "weight": float(self.weights[i]), "contribution": float(contributions[row, i])}
                                   for i, name in enumerate(FEATURE_NAMES)]}
                for row, score in enumerate(raw_scores)]

    def to_dict(self):
        return {"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
                "score_policy_version": SCORE_POLICY_VERSION,
                "feature_fingerprint": feature_fingerprint(),
                "feature_names": FEATURE_NAMES, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "weights": self.weights.tolist(),
                "score_semantics": "Relative linear evidence score plus a declared relevant-experience-shortfall adjustment; not probability or a 0–3 annotation grade",
                "metadata": {**self.metadata, "score_policy_version": SCORE_POLICY_VERSION}}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("model_version") != MODEL_VERSION or data.get("feature_version") != FEATURE_VERSION
                or data.get("score_policy_version") != SCORE_POLICY_VERSION
                or data.get("metadata", {}).get("score_policy_version") != SCORE_POLICY_VERSION
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
        explanations = self.explain([v[0] for v in extracted])
        output = []
        for original, vid, (_, facts), explanation in zip(vacancies, ids, extracted, explanations):
            output.append({**original, "vacancy_id": vid, "ml2_score": explanation["final_score"],
                           "ml2_model": MODEL_VERSION, "ml2_evidence": facts,
                           "ml2_contributions": {row["feature"]: row["contribution"] for row in explanation["contributions"]},
                           "ml2_score_breakdown": {**explanation, "score_policy_version": SCORE_POLICY_VERSION}})
        output.sort(key=lambda r: (-r["ml2_score"], r["vacancy_id"]))
        return output[:top_k]
