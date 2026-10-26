import copy
import json
from pathlib import Path
import tempfile
import unittest

from ML2.prepare_dataset import (REVIEW_CHOICE, build_dataset, candidate_groups,
                                 canonical_quote, parse_task, split_groups)


def task_fixture(number, cv_id, vacancy_id, overall=2, text=None):
    cv = text if text is not None else f"Candidate {cv_id}\nJava\nОпыт работы 1 год"
    vacancy = "Java Developer\n1–3 года\nJava\nPython"
    result = []
    for n, (word, category, quote, score) in enumerate([
        ("Java Developer", "ROLE_LEVEL", "Java", 3),
        ("1–3 года", "EXPERIENCE", "Опыт работы 1 год", 3),
        ("Python", "HARD_SKILL", "", 0),
    ]):
        rid = f"r{n}"
        if not cv:
            quote, score = "", 0
        start = vacancy.index(word)
        result.append({"id": rid, "from_name": "requirement_type", "type": "labels",
                       "value": {"start": start, "end": start + len(word),
                                 "text": word, "labels": [category]}})
        for name, key, value in [("requirement_title", "text", word),
                                  ("reason", "text", "Обоснование по исходному резюме."),
                                  ("match_score", "choices", str(score))]:
            result.append({"id": rid, "from_name": name,
                           "type": "choices" if key == "choices" else "textarea",
                           "value": {key: [value]}})
        if quote:
            result.append({"id": rid, "from_name": "cv_evidence", "type": "textarea",
                           "value": {"text": [quote]}})
    result += [{"id": "overall", "from_name": "overall_score", "type": "choices",
                "value": {"choices": [str(overall)]}},
               {"id": "reason", "from_name": "overall_reason", "type": "textarea",
                "value": {"text": ["Общая оценка соответствия вакансии."]}},
               {"id": "review", "from_name": "human_review", "type": "choices",
                "value": {"choices": [REVIEW_CHOICE]}}]
    return {"id": number, "data": {"pair_id": f"pair_{number}", "cv_id": cv_id,
            "vacancy_id": vacancy_id, "cv_text": cv, "vacancy_display": vacancy,
            "model_assessment": {"overall_score": 3}},
            "annotations": [{"id": number + 1000, "result": result}],
            "predictions": [{"result": [{"from_name": "overall_score",
                                           "value": {"choices": ["3"]}}]}]}


def export_fixture(count=6):
    return [task_fixture(i * 5 + j, f"cv_{i}", f"v_{j}", (i + j) % 3 + 1)
            for i in range(count) for j in range(5)]


class PrepareDatasetTests(unittest.TestCase):
    def test_only_saved_annotation_is_a_target(self):
        row = parse_task(task_fixture(1, "cv", "v", overall=1))
        self.assertEqual((row["overall_score"], row["relevance"]), (1, 0))
        self.assertNotIn("model_assessment", row)
        self.assertNotIn("predictions", row)

    def test_predictions_without_saved_annotation_are_rejected(self):
        task = task_fixture(1, "cv", "v")
        task["annotations"] = []
        with self.assertRaisesRegex(ValueError, "saved annotation"):
            parse_task(task)

    def test_ambiguous_multiple_annotations_are_rejected(self):
        task = task_fixture(1, "cv", "v")
        task["annotations"] *= 2
        with self.assertRaisesRegex(ValueError, "one saved annotation"):
            parse_task(task)

    def test_unknown_is_preserved_separately_from_mismatch(self):
        row = parse_task(task_fixture(1, "empty", "v", overall=0, text=""))
        self.assertIsNone(row["relevance"])
        self.assertFalse(row["judged"])
        self.assertTrue(all(r["status"] == "unknown" and not r["judged"]
                            for r in row["requirements"]))

    def test_quotes_allow_whitespace_changes_but_not_fabricated_words(self):
        source = "Опыт работы\n1\u00a0год"
        self.assertEqual(canonical_quote("Опыт работы 1 год", source), source)
        with self.assertRaisesRegex(ValueError, "source substring"):
            canonical_quote("Опыт работы 5 лет", source)

    def test_nonzero_score_requires_evidence(self):
        task = task_fixture(1, "cv", "v")
        task["annotations"][0]["result"] = [r for r in task["annotations"][0]["result"]
                                             if not (r["from_name"] == "cv_evidence" and r["id"] == "r0")]
        with self.assertRaisesRegex(ValueError, "evidence disagree"):
            parse_task(task)

    def test_invalid_span_and_duplicate_controls_are_rejected(self):
        task = task_fixture(1, "cv", "v")
        broken = copy.deepcopy(task)
        broken["annotations"][0]["result"][0]["value"]["end"] -= 1
        with self.assertRaisesRegex(ValueError, "span text"):
            parse_task(broken)
        task["annotations"][0]["result"].append(copy.deepcopy(task["annotations"][0]["result"][1]))
        with self.assertRaisesRegex(ValueError, "one saved control"):
            parse_task(task)

    def test_bad_source_hash_is_rejected(self):
        task = task_fixture(1, "cv", "v")
        task["data"]["cv_text_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            parse_task(task)

    def test_same_candidate_versions_cannot_cross_splits(self):
        rows = [parse_task(t) for t in export_fixture()]
        groups = candidate_groups(rows, [["cv_0", "cv_1"]])
        self.assertEqual(len(groups), 5)
        splits, assignment = split_groups(rows)
        seen = {}
        for split, part in splits.items():
            for row in part:
                seen.setdefault(row["cv_id"], split)
                self.assertEqual(seen[row["cv_id"]], split)
                self.assertEqual(assignment[row["candidate_group_id"]], split)
        self.assertEqual(seen["cv_0"], seen["cv_1"])
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        reversed_splits, _ = split_groups(list(reversed(rows)))
        self.assertEqual({k: [r["pair_id"] for r in v] for k, v in splits.items()},
                         {k: [r["pair_id"] for r in v] for k, v in reversed_splits.items()})

    def test_identical_cv_versions_are_automatically_grouped(self):
        tasks = export_fixture()
        first_cv = tasks[0]["data"]["cv_text"]
        for task in tasks[5:10]:
            task["data"]["cv_text"] = first_cv
        groups = candidate_groups([parse_task(t) for t in tasks])
        self.assertTrue(any(set(ids) == {"cv_0", "cv_1"} for ids in groups.values()))

    def test_requires_five_distinct_vacancies(self):
        rows = [parse_task(t) for t in export_fixture()[:-1]]
        with self.assertRaisesRegex(ValueError, "five distinct"):
            candidate_groups(rows)

    def test_full_build_quarantines_unknown_and_protects_existing_output(self):
        tasks = export_fixture()
        tasks += [task_fixture(100 + j, "empty", f"v_{j}", overall=0, text="") for j in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "export.json", Path(folder) / "dataset"
            source.write_text(json.dumps(tasks), encoding="utf-8")
            before = source.read_bytes()
            report = build_dataset(source, output, [["cv_0", "cv_1"]])
            self.assertEqual((report["tasks"], report["eligible_pairs"], report["quarantined_pairs"]),
                             (35, 30, 5))
            self.assertEqual(source.read_bytes(), before)
            for split in ("train", "validation", "test"):
                for line in (output / f"{split}.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    self.assertNotEqual(row["overall_score"], 0)
                    self.assertNotIn("requirements", row)
                    self.assertNotIn("overall_explanation", row)
                    self.assertNotIn("empty", row["cv_id"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                build_dataset(source, output)


if __name__ == "__main__":
    unittest.main()
