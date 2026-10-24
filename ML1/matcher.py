import numpy as np
import pandas as pd

from ML1.config import VACANCIES_FILE, EMBEDDINGS_FILE
from ML1.embeddings import encode_text
from ML1.vacancy_index import read_vacancies, load_index
from ML1.skill_similarity import calculate_skill_similarity, get_skill_details
from NLP.skill_extractor import extract_skills


class JobMatcher:

    def __init__(self, alpha=0.4):

        self.alpha = alpha

        print("Loading vacancies...")

        self.vacancies, csv_sha256 = read_vacancies(VACANCIES_FILE)
        print(f"Loaded {len(self.vacancies)} vacancies.")
        print("Loading and validating vacancy embeddings...")
        self.vacancy_embeddings, self.index_metadata = load_index(
            self.vacancies, csv_sha256, EMBEDDINGS_FILE
        )
        print(f"Loaded embeddings: {self.vacancy_embeddings.shape}")

    # --------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------

    def semantic_scores(
        self,
        resume_text
    ):
        """
        Calculate semantic similarity between a resume
        and every vacancy.

        Vacancy embeddings and resume embeddings are
        normalized, so dot product = cosine similarity.
        """

        resume_embedding = encode_text(
            resume_text
        )

        scores = (
            self.vacancy_embeddings
            @ resume_embedding
        )

        # Negative cosine similarities are not useful
        # for the current prototype.
        scores = np.clip(
            scores,
            0.0,
            1.0
        )

        return scores

    # --------------------------------------------------
    # Main matching method
    # --------------------------------------------------

    def match(
        self,
        resume_text,
        top_k=10,
        resume_skills=None
    ):
        """
        Match one resume against all vacancies.

        Parameters
        ----------
        resume_text:
            Full processed resume text.

        top_k:
            Number of highest-ranked vacancies returned.

        resume_skills:
            Optional already-extracted skills.
            If not provided, skills are extracted here.
        """

        print("\nExtracting resume skills...")

        if resume_skills is None:

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

        # --------------------------------------------------
        # Compare resume against every vacancy
        # --------------------------------------------------

        for index, vacancy in (
            self.vacancies.iterrows()
        ):

            vacancy_skills = vacancy.get(
                "skills",
                ""
            )

            if pd.isna(vacancy_skills):
                vacancy_skills = ""

            # Skill similarity
            skill_score = (
                calculate_skill_similarity(
                    resume_skills,
                    vacancy_skills
                )
            )

            # Semantic similarity
            semantic_score = float(
                semantic_scores[index]
            )

            # Hybrid score
            hybrid_score = (
                self.alpha * skill_score
                +
                (1 - self.alpha)
                * semantic_score
            )

            matched, missing = (
                get_skill_details(
                    resume_skills,
                    vacancy_skills
                )
            )

            # --------------------------------------------------
            # Collect complete vacancy context
            # --------------------------------------------------

            result = {

                "index":
                    int(index),

                "vacancy_id":
                    vacancy["vacancy_id"],

                "title":
                    vacancy.get(
                        "title",
                        "Unknown"
                    ),

                "city":
                    vacancy.get(
                        "city",
                        "Unknown"
                    ),

                "job_category":
                    vacancy.get(
                        "Job",
                        "Unknown"
                    ),

                # Full text used by the future assistant
                "description":
                    vacancy.get(
                        "text",
                        ""
                    ),

                "requirements":
                    vacancy.get(
                        "requirements",
                        ""
                    ),

                "responsibilities":
                    vacancy.get(
                        "responsibilities",
                        ""
                    ),

                "experience":
                    vacancy.get(
                        "experience",
                        ""
                    ),

                "employment":
                    vacancy.get(
                        "employment",
                        ""
                    ),

                "salary":
                    vacancy.get(
                        "salary",
                        ""
                    ),

                "url":
                    vacancy.get(
                        "url",
                        ""
                    ),

                # Scores
                "skill_score":
                    skill_score,

                "semantic_score":
                    semantic_score,

                "hybrid_score":
                    hybrid_score,

                # Skill explanation
                "matched_skills":
                    matched,

                "missing_skills":
                    missing,

                "vacancy_skills":
                    vacancy_skills,
            }

            results.append(
                result
            )

        # --------------------------------------------------
        # Rank
        # --------------------------------------------------

        results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:top_k]
