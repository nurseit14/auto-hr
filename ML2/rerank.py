"""Apply the trained model to source CV text and a vacancy shortlist."""

import argparse
import csv
import json
from pathlib import Path

from ML2.ranker import LinearRanker
from ML2.train import ROOT, read_jsonl


def demo_sources(dataset, cv_id):
    rows = [r for r in read_jsonl(Path(dataset) / "pairs_all.jsonl") if r["cv_id"] == cv_id]
    if not rows:
        raise ValueError("CV ID not found in the local dataset")
    return rows[0]["cv_text"], [{**r["vacancy_fields"], "vacancy_id": r["vacancy_id"]} for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Пересортировать вакансии обученным ML2")
    parser.add_argument("--model", type=Path, default=ROOT / "data/annotations/ml2_reranker_v1/model.json")
    parser.add_argument("--cv", type=Path, help="Текстовый файл резюме UTF-8")
    parser.add_argument("--vacancies", type=Path, help="JSON-список или CSV выбранных вакансий / результатов ML1")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/annotations/ml2_training_v1")
    parser.add_argument("--cv-id", help="Пример из локального датасета вместо --cv и --vacancies")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--brief", action="store_true", help="Показать короткий результат для человека вместо JSON")
    args = parser.parse_args(argv)
    if args.cv_id:
        if args.cv or args.vacancies:
            parser.error("Choose --cv-id or --cv with --vacancies")
        cv, vacancies = demo_sources(args.dataset, args.cv_id)
    else:
        if not args.cv or not args.vacancies:
            parser.error("Pass --cv-id, or both --cv and --vacancies")
        cv = args.cv.read_text(encoding="utf-8")
        if args.vacancies.suffix.casefold() == ".csv":
            with args.vacancies.open(encoding="utf-8-sig", newline="") as stream:
                vacancies = list(csv.DictReader(stream))
        else:
            vacancies = json.loads(args.vacancies.read_text(encoding="utf-8"))
    if not isinstance(vacancies, list):
        raise ValueError("Vacancies must be a list")
    ranked = LinearRanker.load(args.model).rerank(cv, vacancies, top_k=args.top_k)
    output = {"experimental": True, "score_note": "Relative ranking score; not a probability or 0–3 grade",
              "results": ranked}
    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.brief:
        lines = ["Порядок вакансий после ML2", ""]
        for index, row in enumerate(ranked, 1):
            lines.append(f"{index}. {row.get('title') or row.get('vacancy_title', '')} (ID {row['vacancy_id']})")
            lines.append(f"   Балл ранжирования: {row['ml2_score']:.3f}")
            matches = ", ".join(row["ml2_evidence"]["confirmed_skill_matches"])
            lines.append("   Совпавшие технологии: " + (matches or "нет подтверждённых совпадений по словарю"))
        lines += ["", "Это экспериментальный порядок вакансий, а не вероятность найма или оценка 0–3."]
        text = "\n".join(lines) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print("Результат:", args.output)
    else:
        print(text)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Ошибка: {error}")
