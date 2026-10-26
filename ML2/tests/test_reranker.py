import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ML2.evaluate import evaluate_saved
from ML2.features import (FEATURE_NAMES, claimed_years, extract_features,
                          language_levels, professional_text, seniority)
from ML2.metrics import evaluate, expected_dcg
from ML2.prepare_dataset import build_dataset
from ML2.ranker import LinearRanker, preference_pairs
from ML2.tests.test_prepare_dataset import export_fixture
from ML2.train import load_dataset, train_dataset


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.cv = "Junior Java Developer\nНавыки: Java, Spring, SQL\nОпыт работы — 1 год"
        self.vacancy = {"vacancy_id": "java", "title": "Middle Java Developer",
                        "experience": "От 1 года до 3 лет", "requirements": "Java, Spring, SQL, Python"}

    def test_source_skills_level_and_inclusive_experience(self):
        vector, facts = extract_features(self.cv, self.vacancy)
        features = dict(zip(FEATURE_NAMES, vector))
        self.assertEqual(features["skill_coverage"], .75)
        self.assertLess(features["seniority_compatibility"], 0)
        self.assertEqual(features["experience_compatibility"], 1.)
        self.assertEqual(facts["skills_not_confirmed"], ["python"])

    def test_contact_demographics_and_annotation_targets_are_not_features(self):
        base = extract_features(self.cv, self.vacancy)[0]
        extra = "\nЖенщина, родилась 1 января 2000\nГражданство: Pythonland\nNationality: SQL\nEmail: person@example.com\nGitHub: https://github.com/person"
        poison = {"vacancy_fields": self.vacancy, "overall_score": 3,
                  "requirements": [{"category": "HARD_SKILL", "score": 3, "requirement": "Python"}],
                  "overall_explanation": "Senior Python expert", "model_assessment": "Python"}
        self.assertEqual(extract_features(self.cv + extra, poison)[0], base)

    def test_placeholder_technology_is_not_evidence(self):
        text = "Java developer\nMachine Learning Project\nDescribe the task, model and tools (e.g. Python, scikit-learn) and result."
        vector, facts = extract_features(text, {"title": "Python Developer", "requirements": "Python"})
        self.assertNotIn("python", facts["confirmed_skill_matches"])
        self.assertNotIn("Machine Learning Project", professional_text(text))

    def test_language_levels_are_required_on_both_sides_and_do_not_bleed(self):
        self.assertEqual(language_levels("English: C1; Russian: A2"), {"english": 5, "russian": 2})
        self.assertEqual(language_levels("Fluent in English and Russian"), {"english": 5, "russian": 5})
        self.assertEqual(language_levels("English\nRussian"), {})
        v, facts = extract_features("Java developer\nEnglish", {"title": "Developer", "requirements": "English B2"})
        self.assertEqual(facts["languages_compared"], [])
        self.assertEqual(v[FEATURE_NAMES.index("language_compatibility")], 0)

    def test_dates_are_not_turned_into_employment_totals(self):
        self.assertIsNone(claimed_years("Student 2024–2027\nDeveloper Present\nInternship 3 months"))
        self.assertAlmostEqual(claimed_years("Work experience — 6 years 5 months"), 6 + 5 / 12)
        self.assertNotIn(4, seniority("Lead international cooperation initiatives"))

    def test_no_experience_is_an_allowance(self):
        vacancy = {**self.vacancy, "experience": "Нет опыта"}
        vector, _ = extract_features(self.cv, vacancy)
        self.assertEqual(vector[FEATURE_NAMES.index("experience_compatibility")], 1.)


class MetricTests(unittest.TestCase):
    def rows(self, labels):
        return [{"cv_id": "one", "relevance": label} for label in labels]

    def test_perfect_reversed_and_tied_order(self):
        rows = self.rows([2, 1, 0])
        self.assertEqual(evaluate(rows, [3, 2, 1])["ndcg_at_5_informative"], 1.)
        self.assertEqual(evaluate(rows, [1, 2, 3])["pairwise_accuracy"], 0.)
        self.assertEqual(evaluate(rows, [1, 1, 1])["pairwise_accuracy"], .5)
        expected = sum(1 / math.log2(i + 2) for i in range(3))
        self.assertAlmostEqual(expected_dcg([2, 1, 0], [1, 1, 1]), expected)
        self.assertEqual(evaluate(rows, [1, 1, 1]), evaluate(list(reversed(rows)), [1, 1, 1]))

    def test_all_weak_queries_are_not_fake_successes(self):
        metrics = evaluate(self.rows([0, 0, 0]), [1, 2, 3])
        self.assertIsNone(metrics["ndcg_at_5"])
        self.assertIsNone(metrics["ndcg_at_5_informative"])
        self.assertEqual(metrics["all_weak_queries_excluded_from_ndcg"], 1)

    def test_unknown_and_nonfinite_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.rows([None, 1]), [1, 2])
        with self.assertRaises(ValueError):
            evaluate(self.rows([0, 1]), [1, float("nan")])


class LearningTests(unittest.TestCase):
    def test_learns_preferences_and_survives_save_load(self):
        rows = [{"cv_id": f"cv{i // 3}", "relevance": i % 3, "overall_score": i % 3 + 1}
                for i in range(9)]
        x = np.zeros((9, len(FEATURE_NAMES)))
        x[:, 0] = [r["relevance"] / 2 for r in rows]
        model = LinearRanker.fit(rows, x)
        scores = model.predict(x)
        self.assertTrue(scores[0] < scores[1] < scores[2])
        self.assertTrue(np.all(model.weights >= 0))
        self.assertLess(model.metadata["training_objective"], math.log(2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            model.save(path)
            np.testing.assert_array_equal(LinearRanker.load(path).predict(x), scores)

    def test_unknown_labels_cannot_train(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            preference_pairs([{"cv_id": "one", "relevance": None, "overall_score": 0}],
                             np.zeros((1, len(FEATURE_NAMES))))

    def test_ml1_adapter_preserves_sources_ids_and_original_scores(self):
        weights = np.zeros(len(FEATURE_NAMES)); weights[0] = 1
        model = LinearRanker(np.zeros(len(weights)), np.ones(len(weights)), weights)
        vacancies = [{"vacancy_id": "python", "title": "Python developer", "requirements": "Python", "hybrid_score": .9},
                     {"vacancy_id": "java", "title": "Java developer", "requirements": "Java", "hybrid_score": .4}]
        before = copy.deepcopy(vacancies)
        result = model.rerank("Junior Java Developer\nSkills: Java", vacancies)
        self.assertEqual([r["vacancy_id"] for r in result], ["java", "python"])
        self.assertEqual(result[0]["hybrid_score"], .4)
        self.assertEqual(vacancies, before)
        with self.assertRaises(ValueError):
            model.rerank("", vacancies)
        with self.assertRaises(ValueError):
            model.rerank("Java", [vacancies[0], vacancies[0]])

    def test_build_train_evaluate_and_leakage_guard(self):
        tasks = export_fixture()
        for task in tasks:
            task["data"]["vacancy_fields"] = {"title": "Java Developer", "experience": "1–3 года", "requirements": "Java, Python"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, dataset = root / "export.json", root / "dataset"
            source.write_text(json.dumps(tasks), encoding="utf-8")
            build_dataset(source, dataset)
            output = root / "model"
            report = train_dataset(dataset, output)
            reproduced = evaluate_saved(output / "model.json", dataset)
            self.assertEqual(report["test"], reproduced)
            with self.assertRaisesRegex(ValueError, "not empty"):
                train_dataset(dataset, output)
            frozen = dataset / "test.jsonl"
            records = [json.loads(line) for line in frozen.read_text().splitlines()]
            records[0]["candidate_group_id"] = "fake-group"
            frozen.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_dataset(dataset)
            with self.assertRaisesRegex(ValueError, "changed"):
                evaluate_saved(output / "model.json", dataset)


if __name__ == "__main__":
    unittest.main()
