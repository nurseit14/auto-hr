from pathlib import Path

import numpy as np
import pandas as pd

from ML1.embeddings import encode_text
from ML1.skill_similarity import (
    calculate_skill_similarity,
    get_skill_details,
)
from NLP.skill_extractor import extract_skills


ROOT_DIR = Path(__file__).resolve().parent.parent

VACANCIES_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "vacancies_with_skills.csv"
)

EMBEDDINGS_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "vacancy_embeddings.npy"
)


class JobMatcher:

    def __init__(self, alpha=0.6):

        self.alpha = alpha

        print("Loading vacancies...")

        self.vacancies = pd.read_csv(
            VACANCIES_FILE
        )

        print(
            f"Loaded {len(self.vacancies)} vacancies."
        )

        print("Loading vacancy embeddings...")

        self.vacancy_embeddings = np.load(
            EMBEDDINGS_FILE
        )

        if len(self.vacancies) != len(
            self.vacancy_embeddings
        ):
            raise ValueError(
                "Vacancy dataset and embeddings "
                "have different sizes."
            )

        print(
            f"Loaded embeddings: "
            f"{self.vacancy_embeddings.shape}"
        )

    def semantic_scores(
        self,
        resume_text
    ):
        """
        Calculate semantic similarity between
        resume and all vacancies.
        """

        resume_embedding = encode_text(
            resume_text
        )

        # Embeddings are normalized, therefore
        # dot product = cosine similarity
        scores = (
            self.vacancy_embeddings
            @ resume_embedding
        )

        # Cosine similarity can theoretically
        # be negative.
        scores = np.clip(
            scores,
            0.0,
            1.0
        )

        return scores

    def match(
        self,
        resume_text,
        top_k=10
    ):

        print("\nExtracting resume skills...")

        resume_skills = extract_skills(
            resume_text
        )

        print(
            "Detected resume skills:",
            resume_skills
        )

        print(
            "\nCalculating semantic similarity..."
        )

        semantic_scores = self.semantic_scores(
            resume_text
        )

        results = []

        for index, vacancy in (
            self.vacancies.iterrows()
        ):

            vacancy_skills = vacancy.get(
                "skills",
                ""
            )

            skill_score = (
                calculate_skill_similarity(
                    resume_skills,
                    vacancy_skills
                )
            )

            semantic_score = float(
                semantic_scores[index]
            )

            hybrid_score = (
                self.alpha * skill_score
                + (1 - self.alpha)
                * semantic_score
            )

            matched, missing = (
                get_skill_details(
                    resume_skills,
                    vacancy_skills
                )
            )

            results.append(
                {
                    "index": index,

                    "title": vacancy.get(
                        "title",
                        "Unknown"
                    ),

                    "city": vacancy.get(
                        "city",
                        "Unknown"
                    ),

                    "job_category": vacancy.get(
                        "Job",
                        "Unknown"
                    ),

                    "skill_score": skill_score,

                    "semantic_score":
                        semantic_score,

                    "hybrid_score":
                        hybrid_score,

                    "matched_skills":
                        matched,

                    "missing_skills":
                        missing,

                    "vacancy_skills":
                        vacancy_skills,
                }
            )

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:top_k]
