"""Apply the fixed ML2-04 baseline to an already ordered ML1 shortlist.

This command does not execute ML1, download an encoder, open trained weights,
or use annotations. The packaged demo contains explicitly fictional scores.
"""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from ML2.baseline_ranker import BASELINE_VERSION, BASELINE_WEIGHTS, TransparentBaseline
from ML2.features import FEATURE_VERSION


FIXTURES = Path(__file__).with_name("tests") / "fixtures"
SCORE_FIELDS = ("skill_score", "semantic_score", "hybrid_score")
SCORE_NOTE = "Относительный балл фиксированной формулы; не вероятность найма и не оценка разметки 0–3."


def invalid_constant(value):
    raise ValueError(f"JSON содержит недопустимое число {value}")


def load_vacancies(path):
    """Read an ML1-like list; CSV score cells are adapted to numeric JSON values."""
    path = Path(path)
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for number, row in enumerate(rows, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV, строка {number}: число значений не совпадает с заголовком")
            for field in SCORE_FIELDS:
                if field in row:
                    try:
                        row[field] = float(row[field])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"CSV, строка {number}: поле {field} должно содержать число") from exc
            if "matching_metadata" in row and row["matching_metadata"].strip():
                try:
                    metadata = json.loads(row["matching_metadata"], parse_constant=invalid_constant)
                except ValueError as exc:
                    raise ValueError(f"CSV, строка {number}: matching_metadata должен содержать JSON-объект") from exc
                if not isinstance(metadata, dict):
                    raise ValueError(f"CSV, строка {number}: matching_metadata должен содержать JSON-объект")
                row["matching_metadata"] = metadata
    elif path.suffix.casefold() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)
    else:
        raise ValueError("Файл вакансий должен иметь расширение .json или .csv")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Вакансии должны быть списком объектов в исходном порядке ML1")
    return rows


def initial_order(vacancies):
    """Keep the complete input order and source scores even when output is Top-N."""
    output = []
    for rank, row in enumerate(vacancies, 1):
        entry = {
            "ml1_rank": rank,
            "vacancy_id": str(row.get("vacancy_id", row.get("id", ""))),
            "title": row.get("title", row.get("vacancy_title", "")),
        }
        for field in (*SCORE_FIELDS, "matching_metadata"):
            if field in row:
                entry[field] = deepcopy(row[field])
        output.append(entry)
    return output


def brief_result(output):
    lines = ["ML2-04: прозрачный baseline", ""]
    if output["demo_input"]:
        lines += ["Демонстрационные вымышленные данные. Баллы ML1 заданы вручную; реальный поиск ML1 не запускался.", ""]
    lines += ["До пересортировки ML1:"]
    for row in output["ml1_initial_order"]:
        lines.append(f"  {row['ml1_rank']}. {row['title']} (ID {row['vacancy_id']}), hybrid_score={row['hybrid_score']:.4f}")
    lines += ["", "После фиксированной формулы:"]
    for row in output["results"]:
        lines.append(f"  {row['baseline_rank']}. {row.get('title') or row.get('vacancy_title', '')} (ID {row['vacancy_id']})")
        lines.append(f"     Исходное место ML1: {row['ml1_rank']}; балл baseline: {row['baseline_score']:+.4f}")
        for component in row["baseline_breakdown"]["components"]:
            lines.append(f"     {component['label']}: {component['contribution']:+.4f} ({component['status']})")
    lines += ["", SCORE_NOTE, "Веса заданы заранее и не обучались. Проверка качества ожидает завершения ML2-03."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Пересортировать готовую выдачу ML1 фиксированным baseline ML2-04")
    parser.add_argument("--cv", type=Path, help="Исходный текст резюме UTF-8")
    parser.add_argument("--vacancies", type=Path, help="JSON-список или CSV результатов ML1 с hybrid_score")
    parser.add_argument("--demo", action="store_true", help="Вымышленные данные с заданными вручную баллами ML1")
    parser.add_argument("--top-k", type=int, default=10, help="Число вакансий в итоговой выдаче, по умолчанию 10")
    parser.add_argument("--output", type=Path, help="Новый файл JSON; существующий файл не перезаписывается")
    parser.add_argument("--brief", action="store_true", help="Показать до/после и вклады по-русски; --output сохраняет JSON")
    args = parser.parse_args(argv)
    if args.demo:
        if args.cv or args.vacancies:
            parser.error("Выберите --demo либо --cv вместе с --vacancies")
        cv_path, vacancies_path = FIXTURES / "baseline_cv.txt", FIXTURES / "baseline_vacancies.json"
    else:
        if args.cv is None or args.vacancies is None:
            parser.error("Передайте --demo либо оба параметра --cv и --vacancies")
        cv_path, vacancies_path = args.cv, args.vacancies
    cv = cv_path.read_text(encoding="utf-8-sig")
    vacancies = load_vacancies(vacancies_path)
    ranked = TransparentBaseline().rerank(cv, vacancies, top_k=args.top_k)
    fixture_input = args.demo or bool(vacancies) and all(
        isinstance(row.get("matching_metadata"), dict) and row["matching_metadata"].get("fixture") is True
        for row in vacancies
    )
    output = {
        "baseline_version": BASELINE_VERSION,
        "feature_version": FEATURE_VERSION,
        "quality_validation": "pending_ml2_03",
        "weights": dict(BASELINE_WEIGHTS),
        "score_note": SCORE_NOTE,
        "demo_input": fixture_input,
        "input_note": (
            "Fictional fixture; supplied ML1-like scores were not produced by an executed ML1 search"
            if fixture_input else "Caller-supplied ML1 shortlist; this command does not run retrieval"
        ),
        "input_count": len(vacancies),
        "output_count": len(ranked),
        "ml1_initial_order": initial_order(vacancies),
        "results": ranked,
    }
    # Serialize before opening the destination: invalid input cannot leave a
    # misleading partial result, and strict JSON does not admit NaN or Infinity.
    serialized = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    if args.brief:
        print(brief_result(output), end="")
        if args.output:
            print(f"JSON сохранён: {args.output}")
    elif args.output:
        print(f"Результат: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"Ошибка: {error}")
