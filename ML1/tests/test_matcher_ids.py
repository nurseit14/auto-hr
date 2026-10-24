"""Vacancy identity regressions using real CSV parsing and a fake encoder.

Run: python -m unittest discover -s ML1/tests -v
No model download, API, real CVs or prebuilt embedding file is required.
"""

import csv
import importlib.util
import json
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


# Load an isolated module so this test's encoder stub does not affect other tests
# or import sentence-transformers/torch when merely checking source IDs.
SPEC = importlib.util.spec_from_file_location(
    "matcher_identity_under_test", Path(__file__).resolve().parents[1] / "matcher.py"
)
matcher_module = importlib.util.module_from_spec(SPEC)
encoder_module = types.ModuleType("ML1.embeddings")
encoder_module.encode_text = Mock(return_value=np.array([1.0, 0.0]))
with patch.dict("sys.modules", {"ML1.embeddings": encoder_module}):
    SPEC.loader.exec_module(matcher_module)


class VacancyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="autohr_vacancy_ids_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    @staticmethod
    def row(vacancy_id, title, **extra):
        return {
            "vacancy_id": vacancy_id,
            "title": title,
            "text": title + " requirements",
            "skills": "python, docker",
            "url": "https://example.invalid/" + str(vacancy_id),
            **extra,
        }

    def create(self, rows, vectors=None):
        path = self.directory / "vacancies.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if vectors is None:
            vectors = np.tile([1.0, 0.0], (len(rows), 1))
        embedding_path = self.directory / "embeddings.npy"
        np.save(embedding_path, vectors)
        with ExitStack() as stack:
            stack.enter_context(patch.object(matcher_module, "VACANCIES_FILE", path))
            stack.enter_context(patch.object(matcher_module, "EMBEDDINGS_FILE", embedding_path))
            stack.enter_context(redirect_stdout(StringIO()))
            return matcher_module.JobMatcher(alpha=0.4)

    @staticmethod
    def match(matcher):
        with redirect_stdout(StringIO()):
            return matcher.match("fictional Python CV", top_k=10, resume_skills=["python", "docker"])

    def test_real_id_wins_over_legacy_id_and_row_position(self):
        rows = [self.row("job-12", "Python Backend", id="wrong-id-0"),
                self.row("job-77", "Python Data", id="wrong-id-1")]
        results = self.match(self.create(rows))
        self.assertEqual([r["vacancy_id"] for r in results], ["job-12", "job-77"])
        self.assertEqual([r["index"] for r in results], [0, 1])

    def test_ids_are_exact_strings_including_zeros_large_numbers_and_na_tokens(self):
        ids = ["0012", "9007199254740993", "1e3", "NA", "null", "job-003"]
        results = self.match(self.create([self.row(vid, "Job " + vid) for vid in ids]))
        self.assertEqual([r["vacancy_id"] for r in results], ids)
        self.assertTrue(all(isinstance(r["vacancy_id"], str) for r in results))
        self.assertEqual([r["vacancy_id"] for r in json.loads(json.dumps(results))], ids)

    def test_ranked_context_stays_with_its_source_id(self):
        rows = [self.row("python-42", "Python Backend", requirements="Python APIs"),
                self.row("other-91", "Other Vacancy", requirements="Other requirements")]
        results = self.match(self.create(rows, np.array([[0.0, 1.0], [1.0, 0.0]])))
        self.assertEqual([r["vacancy_id"] for r in results], ["other-91", "python-42"])
        for result in results:
            source = next(row for row in rows if row["vacancy_id"] == result["vacancy_id"])
            self.assertEqual(result["title"], source["title"])
            self.assertEqual(result["description"], source["text"])
            self.assertEqual(result["requirements"], source["requirements"])
            self.assertEqual(result["url"], source["url"])
            self.assertEqual(result["vacancy_skills"], source["skills"])

    def test_reordering_rows_and_their_vectors_preserves_identity_and_scores(self):
        # Reordering CSV WITHOUT its matching vectors is ML1-03, not this fix.
        rows = [self.row("job-a", "First"), self.row("job-b", "Second")]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        original = self.match(self.create(rows, vectors))
        reordered = self.match(self.create(rows[::-1], vectors[::-1]))
        self.assertEqual([r["vacancy_id"] for r in original], [r["vacancy_id"] for r in reordered])
        self.assertEqual([r["hybrid_score"] for r in original], [r["hybrid_score"] for r in reordered])
        self.assertEqual(original[0]["index"], 0)
        self.assertEqual(reordered[0]["index"], 1)

    def assert_bad_csv(self, rows, message):
        with patch.object(matcher_module.np, "load") as load:
            with self.assertRaisesRegex(ValueError, message):
                self.create(rows)
            load.assert_not_called()

    def test_missing_id_column_cannot_fall_back_to_legacy_id(self):
        self.assert_bad_csv([{"id": "legacy-1", "title": "Job", "skills": "python"}], "vacancy_id.*column")

    def test_blank_and_whitespace_only_ids_fail_before_index_loading(self):
        for vid in ("", " ", "\t"):
            with self.subTest(vid=repr(vid)):
                self.assert_bad_csv([self.row(vid, "Job")], "empty vacancy_id")

    def test_duplicate_ids_fail_before_index_loading(self):
        self.assert_bad_csv([self.row("same", "First"), self.row("same", "Second")], "duplicate vacancy_id")

    def test_outer_whitespace_is_not_silently_normalized(self):
        for vid in (" job-1", "job-1 "):
            with self.subTest(vid=repr(vid)):
                self.assert_bad_csv([self.row(vid, "Job")], "whitespace")

    def test_distinct_source_ids_are_not_coerced_to_the_same_number(self):
        results = self.match(self.create([self.row("001", "First"), self.row("1", "Second")]))
        self.assertEqual([r["vacancy_id"] for r in results], ["001", "1"])

    def test_embedding_count_check_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "different sizes"):
            self.create([self.row("job-1", "Job")], np.array([[1.0, 0.0], [0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
