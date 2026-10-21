import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

from ML2.data import load_resumes, select_pairs
from ML2.demo import demo_inputs
from ML2.label_studio import make_task, publish
from ML2.model import ML2Annotator
from ML2.schemas import (PairAssessment, ModelPairAssessment, ground_model_answer,
                         numbered_lines, validate_assessment)


def example_answer():
    return PairAssessment.model_validate({
        "requirements": [
            {"category": "ROLE_LEVEL", "requirement": "Middle Java разработчик",
             "vacancy_quote": "Middle Java разработчик", "score": 1, "status": "mismatch",
             "cv_quote": "Junior Java разработчик", "explanation": "Junior ниже требуемого Middle."},
            {"category": "EXPERIENCE", "requirement": "Опыт 1–3 года", "vacancy_quote": "1–3 года",
             "score": 3, "status": "match", "cv_quote": "Опыт работы 1 год", "explanation": "1 год входит в диапазон."},
            *[{"category": "HARD_SKILL", "requirement": name, "vacancy_quote": name,
               "score": 3, "status": "match", "cv_quote": name, "explanation": "Навык прямо указан."}
              for name in ["Java", "Spring", "SQL"]],
            {"category": "HARD_SKILL", "requirement": "Python", "vacancy_quote": "Python",
             "score": 0, "status": "unknown", "cv_quote": "", "explanation": "Сведений о Python нет."},
        ], "overall_score": 2, "overall_explanation": "Есть совпадение стека и опыта, но уровень Junior ниже Middle.",
        "warnings": [],
    })


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.cvs, self.vacancies = demo_inputs()
        self.vacancy = next(v for v in self.vacancies if v["vacancy_id"] == "demo_middle")
        self.pair = next(p for p in select_pairs(self.cvs, self.vacancies) if p["vacancy"]["vacancy_id"] == "demo_middle")

    def test_user_example_and_inclusive_experience(self):
        answer = example_answer()
        validate_assessment(answer, self.cvs[0]["text"], self.vacancy)
        self.assertEqual([r.score for r in answer.requirements], [1, 3, 3, 3, 3, 0])
        self.assertEqual(answer.overall_score, 2)

    def test_unknown_is_not_a_negative_example(self):
        row = example_answer().requirements[-1]
        self.assertEqual((row.score, row.status, row.cv_quote), (0, "unknown", ""))

    def test_quotes_are_copied_from_selected_source_lines(self):
        cv = "Java\r\n\r\nОпыт работы 1 год\nSpring\nSQL"
        data = example_answer().model_dump()
        for row in data["requirements"]:
            row.pop("cv_quote")
            row["cv_evidence_line_ids"] = [1, 3] if row["score"] else []
            row["vacancy_evidence_line_id"] = next(line["line_id"] for line in numbered_lines(self.vacancy["display"])
                                                   if row["vacancy_quote"] in line["text"])
        answer = ground_model_answer(ModelPairAssessment.model_validate(data), cv, self.vacancy["display"])
        self.assertEqual(answer.requirements[0].cv_quote, "Java\r\n\r\nОпыт работы 1 год")
        validate_assessment(answer, cv, self.vacancy)

    def test_invalid_evidence_line_is_rejected(self):
        data = example_answer().model_dump()
        for row in data["requirements"]:
            row.pop("cv_quote")
            row["cv_evidence_line_ids"] = [99999] if row["score"] else []
            row["vacancy_evidence_line_id"] = 2
        with self.assertRaisesRegex(ValueError, "does not exist"):
            ground_model_answer(ModelPairAssessment.model_validate(data), self.cvs[0]["text"], self.vacancy["display"])

    def test_fabricated_cv_evidence_is_rejected(self):
        answer = example_answer()
        answer.requirements[1].cv_quote = "Опыт работы 5 лет"
        with self.assertRaisesRegex(ValueError, "exact CV"):
            validate_assessment(answer, self.cvs[0]["text"], self.vacancy)

    def test_fabricated_vacancy_requirement_is_rejected(self):
        answer = example_answer()
        answer.requirements[-1].vacancy_quote = "Docker Kubernetes"
        with self.assertRaisesRegex(ValueError, "source substring"):
            validate_assessment(answer, self.cvs[0]["text"], self.vacancy)

    def test_absence_cannot_receive_mismatch_without_evidence(self):
        answer = example_answer()
        answer.requirements[-1].score, answer.requirements[-1].status = 1, "mismatch"
        with self.assertRaisesRegex(ValueError, "require direct CV evidence"):
            validate_assessment(answer, self.cvs[0]["text"], self.vacancy)

    def test_empty_cv_only_allows_unknown(self):
        answer = example_answer()
        with self.assertRaises(ValueError):
            validate_assessment(answer, "", self.vacancy)
        for row in answer.requirements:
            row.score, row.status, row.cv_quote = 0, "unknown", ""
        answer.overall_score = 0
        validate_assessment(answer, "", self.vacancy)

    def test_missing_experience_assessment_is_rejected(self):
        answer = example_answer()
        answer.requirements = [r for r in answer.requirements if r.category != "EXPERIENCE"]
        with self.assertRaisesRegex(ValueError, "experience"):
            validate_assessment(answer, self.cvs[0]["text"], self.vacancy)

    def test_role_mismatch_caps_overall_score(self):
        answer = example_answer()
        answer.overall_score = 3
        with self.assertRaisesRegex(ValueError, "role/level mismatch"):
            validate_assessment(answer, self.cvs[0]["text"], self.vacancy)

    def test_exactly_five_unique_pairs_and_stable_ids(self):
        cvs = [{**self.cvs[0], "cv_id": f"cv_{i}"} for i in range(44)]
        pairs = select_pairs(cvs, self.vacancies)
        self.assertEqual(len(pairs), 220)
        self.assertEqual(set(Counter(p["cv"]["cv_id"] for p in pairs).values()), {5})
        self.assertEqual(len({p["pair_id"] for p in pairs}), 220)
        shuffled = select_pairs(cvs, list(reversed(self.vacancies)))
        self.assertEqual([p["pair_id"] for p in pairs], [p["pair_id"] for p in shuffled])

    def test_empty_resume_remains_in_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resumes.csv"
            path.write_text("resume_id,text,skills\nempty,,\n", encoding="utf-8")
            rows = load_resumes(path)
            self.assertEqual(len(select_pairs(rows, self.vacancies)), 5)

    def test_not_enough_distinct_vacancies_fails(self):
        with self.assertRaises(ValueError):
            select_pairs(self.cvs, self.vacancies[:4])

    def test_duplicate_resume_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resumes.csv"
            path.write_text("resume_id,text\nsame,Java\nsame,Python\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate CV"):
                load_resumes(path)

    def test_label_studio_spans_and_controls_match_config(self):
        answer = example_answer()
        record = {"assessment": answer.model_dump(), "model_version": "test-model"}
        task = make_task(self.pair, record, "test-dataset")
        root = ET.parse(Path(__file__).parents[1] / "label_config.xml").getroot()
        names = {el.attrib["name"] for el in root.iter() if "name" in el.attrib}
        result = task["predictions"][0]["result"]
        for entry in result:
            self.assertIn(entry["from_name"], names)
            self.assertIn(entry["to_name"], names)
            if entry["type"] == "labels":
                value = entry["value"]
                self.assertEqual(self.vacancy["display"][value["start"]:value["end"]], value["text"])
                region_entries = [r for r in result if r["id"] == entry["id"]]
                self.assertIn("match_score", [r["from_name"] for r in region_entries])
        self.assertNotIn("human_review", [r["from_name"] for r in result])
        self.assertNotIn("annotations", task)
        self.assertEqual(next(r for r in result if r["from_name"] == "overall_score")["value"]["choices"], ["2"])

    def test_failed_task_has_no_fake_prediction(self):
        task = make_task(self.pair, None, "test", "Model unavailable")
        self.assertEqual(task["predictions"], [])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Finish all"):
                publish([task], Path(directory), "http://localhost:8080", "test", "test")

    def test_ready_task_does_not_keep_a_pending_warning(self):
        record = {"assessment": example_answer().model_dump(), "model_version": "test-model"}
        task = make_task(self.pair, record, "test", "Ожидает обработки")
        self.assertNotIn("Ожидает обработки", task["data"]["warnings_text"])
        self.assertEqual(task["data"]["preannotation_status"], "ready")

    def test_model_cache_avoids_second_api_call(self):
        model = ML2Annotator.__new__(ML2Annotator)
        model.model, model.reference_date, model.version = "test", "2026-10-21", "test-v1"
        answer = example_answer().model_dump()
        for row in answer["requirements"]:
            row["cv_evidence_line_ids"] = [1] if row.pop("cv_quote") else []
            row["vacancy_evidence_line_id"] = next(line["line_id"] for line in numbered_lines(self.vacancy["display"])
                                                   if row["vacancy_quote"] in line["text"])
        response = Mock(output_parsed=ModelPairAssessment.model_validate(answer), id="response_test", usage=None)
        model.client = Mock()
        model.client.responses.parse.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            first = model.annotate(self.pair, Path(directory))
            second = model.annotate(self.pair, Path(directory))
            self.assertFalse(first["from_cache"])
            self.assertTrue(second["from_cache"])
            self.assertEqual(model.client.responses.parse.call_count, 1)
            changed = copy.deepcopy(self.pair)
            changed["cv"]["text"] += " Дополнение."
            model.annotate(changed, Path(directory))
            self.assertEqual(model.client.responses.parse.call_count, 2)


if __name__ == "__main__":
    unittest.main()
