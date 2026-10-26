"""Shared configuration for vacancy indexing and query embeddings."""
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
VACANCIES_FILE = ROOT_DIR / "data/processed/vacancies_with_skills.csv"
EMBEDDINGS_FILE = ROOT_DIR / "data/processed/vacancy_embeddings.npy"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# Baseline ranking settings shared by both CLIs and the matcher.
DEFAULT_ALPHA = 0.4
DEFAULT_CONFIDENCE_TARGET = 5.0
DEFAULT_TOP_K = 10
ZERO_SKILL_POLICY = "zero_skill_score_fixed_weights"


def validate_positive_number(value, name):
    if (
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value) or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


def validate_top_k(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("top_k must be a positive integer.")


@dataclass(frozen=True)
class MatchingConfig:
    alpha: float = DEFAULT_ALPHA
    confidence_target: float = DEFAULT_CONFIDENCE_TARGET
    top_k: int = DEFAULT_TOP_K

    def __post_init__(self):
        if (
            isinstance(self.alpha, bool) or not isinstance(self.alpha, Real)
            or not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1
        ):
            raise ValueError("alpha must be a finite number between 0 and 1.")
        validate_positive_number(self.confidence_target, "confidence_target")
        validate_top_k(self.top_k)

    def metadata(self):
        return {
            "alpha": float(self.alpha),
            "semantic_weight": float(1 - self.alpha),
            "confidence_target": float(self.confidence_target),
            "top_k": int(self.top_k),
            "zero_skill_policy": ZERO_SKILL_POLICY,
        }
