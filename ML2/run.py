"""One command: select pairs, run ML2, and publish predictions for human review."""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from ML2.data import digest, load_resumes, load_vacancies, select_pairs
from ML2.label_studio import make_task, publish
from ML2.model import DEFAULT_MODEL, ML2Annotator, save_json
from ML2.prompts import PROMPT_VERSION

ROOT = Path(__file__).resolve().parent.parent


def parser():
    p = argparse.ArgumentParser(description="ML2: автоматическая предразметка по 5 вакансий на резюме")
    p.add_argument("--resumes", type=Path, default=ROOT / "data/resumes/processed/resumes.csv")
    p.add_argument("--vacancies", type=Path, default=ROOT / "data/processed/vacancies_with_skills.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/annotations/ml2_auto_v1")
    p.add_argument("--model", default=os.getenv("ML2_MODEL", DEFAULT_MODEL))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit-resumes", type=int)
    p.add_argument("--demo", action="store_true", help="Использовать вымышленные данные")
    p.add_argument("--prepare-only", action="store_true", help="Только собрать пары, без API")
    p.add_argument("--publish", action="store_true", help="Загрузить в отдельный проект Label Studio")
    p.add_argument("--publish-only", action="store_true", help="Загрузить готовые predictions без вызова модели")
    p.add_argument("--label-studio-url", default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"))
    p.add_argument("--project-title", default="ML2 — 5 вакансий на резюме")
    p.add_argument("--reference-date", help="Дата для расчёта стажа; при продолжении берётся из manifest.json")
    return p


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    args = parser().parse_args(argv)
    if args.workers < 1 or args.workers > 10:
        raise ValueError("workers must be between 1 and 10")
    if args.limit_resumes is not None and args.limit_resumes < 1:
        raise ValueError("limit-resumes must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    args.reference_date = (args.reference_date or (previous_manifest or {}).get("reference_date")
                           or datetime.now(ZoneInfo("Asia/Almaty")).date().isoformat())
    if args.publish_only:
        tasks = json.loads((output / "tasks_with_predictions.json").read_text(encoding="utf-8"))
        version = tasks[0]["predictions"][0]["model_version"]
        state = publish(tasks, output, args.label_studio_url, args.project_title, version)
        print("Label Studio:", state["project_url"], flush=True)
        return 0
    if args.demo:
        from ML2.demo import demo_inputs
        resumes, vacancies = demo_inputs()
    else:
        resumes, vacancies = load_resumes(args.resumes), load_vacancies(args.vacancies)
    if args.limit_resumes:
        resumes = resumes[:args.limit_resumes]
    pairs = select_pairs(resumes, vacancies, count=5)
    dataset_version = "ml2_auto_" + digest({"pairs": [p["pair_id"] for p in pairs],
                                          "prompt": PROMPT_VERSION, "model": args.model,
                                          "reference_date": args.reference_date})[:16]
    manifest = {"dataset_version": dataset_version, "resume_count": len(resumes),
                "vacancy_count": len(vacancies), "pair_count": len(pairs), "pairs_per_resume": 5,
                "empty_cv_ids": [r["cv_id"] for r in resumes if not r["text"].strip()],
                "selection_method": "bm25-skills-v1", "model": args.model,
                "prompt_version": PROMPT_VERSION, "reference_date": args.reference_date,
                "resumes_source": "fictional_demo" if args.demo else str(args.resumes.resolve()),
                "vacancies_source": "fictional_demo" if args.demo else str(args.vacancies.resolve()),
                "pairs": [{"pair_id": p["pair_id"], "cv_id": p["cv"]["cv_id"],
                           "vacancy_id": p["vacancy"]["vacancy_id"], "selection_rank": p["selection_rank"],
                           "selection_score": p["selection_score"], "cv_hash": p["cv"]["text_sha256"],
                           "vacancy_hash": p["vacancy"]["source_sha256"]} for p in pairs]}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["dataset_version"] != dataset_version:
            raise ValueError("Output folder belongs to different data/settings. Choose a new --output-dir.")
    save_json(manifest_path, manifest)
    print(f"Резюме: {len(resumes)}; вакансий: {len(vacancies)}; пар: {len(pairs)}", flush=True)
    if manifest["empty_cv_ids"]:
        print("Пустые CV сохраняются с unknown:", ", ".join(manifest["empty_cv_ids"]), flush=True)
    if args.prepare_only:
        print("Пары подготовлены:", manifest_path, flush=True)
        return 0
    model = ML2Annotator(model=args.model, reference_date=args.reference_date)
    cache_dir = output / "cache"
    records, failures = {}, {}
    def write_tasks():
        tasks = [make_task(p, records.get(p["pair_id"]), dataset_version, failures.get(p["pair_id"], "Ожидает обработки"))
                 for p in pairs]
        save_json(output / "tasks_with_predictions.json", tasks)
        save_json(output / "failures.json", failures)
        return tasks
    write_tasks()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(model.annotate, pair, cache_dir): pair for pair in pairs}
        for completed, future in enumerate(as_completed(futures), 1):
            pair = futures[future]
            try:
                record = future.result()
                records[pair["pair_id"]] = record
                answer = record["assessment"]
                print(f"[{completed}/{len(pairs)}] {pair['cv']['cv_id']} / {pair['vacancy']['vacancy_id']}: "
                      f"требований {len(answer['requirements'])}, итог {answer['overall_score']}"
                      + (" (кэш)" if record["from_cache"] else ""), flush=True)
            except Exception as error:
                message = str(error) if isinstance(error, ValueError) else type(error).__name__
                failures[pair["pair_id"]] = message
                print(f"[{completed}/{len(pairs)}] ОШИБКА {pair['pair_id']}: {message}", flush=True)
            write_tasks()
    tasks = write_tasks()
    scores = Counter(str(r["assessment"]["overall_score"]) for r in records.values())
    requirement_scores = Counter(str(req["score"]) for r in records.values()
                                 for req in r["assessment"]["requirements"])
    summary = {"dataset_version": dataset_version, "total_pairs": len(pairs), "ready": len(records),
               "failed": len(failures), "all_pairs_have_predictions": len(records) == len(pairs),
               "overall_scores": dict(scores), "requirement_scores": dict(requirement_scores),
               "requirements_total": sum(requirement_scores.values()), "empty_cv_ids": manifest["empty_cv_ids"],
               "usage": {key: sum(record["usage"].get(key, 0) for record in records.values())
                         for key in ["input_tokens", "output_tokens", "total_tokens"]},
               "human_review_required": True}
    save_json(output / "summary.json", summary)
    (output / "label_config.xml").write_text(Path(__file__).with_name("label_config.xml").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        print("Есть необработанные пары. Повтори ту же команду: готовые ответы возьмутся из кэша.", flush=True)
        return 1
    if args.publish:
        state = publish(tasks, output, args.label_studio_url, args.project_title, model.version)
        print("Label Studio:", state["project_url"], flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print("Ошибка:", error, file=sys.stderr)
        raise SystemExit(1)
