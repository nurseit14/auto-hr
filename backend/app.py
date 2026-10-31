"""Auto-HR local MVP API. Run with ``uvicorn backend.app:app``."""
from datetime import datetime, timezone
import csv
from io import StringIO
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Literal
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.documents import extract_resume, validate_text, ResumeError, MAX_FILE_BYTES, MAX_TEXT_CHARS
from backend.matching import Matching, CATALOGUE_NOTE, normalize_skill, public_candidate, refresh_skills
from ML2.features import professional_text
from backend.storage import Store
from NLP.anonymizer import anonymize_text
from NLP.skill_extractor import extract_skills

ROOT = Path(__file__).resolve().parent.parent
COOKIE = "autohr_session"


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CandidateInput(Input):
    name: str = Field(default="Кандидат", min_length=1, max_length=120)
    text: str = Field(min_length=20, max_length=MAX_TEXT_CHARS)


class CandidatePatch(Input):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    skills: list[str] | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, values):
        if values is None:
            return None
        if any(not value.strip() or len(value) > 60 or "\n" in value for value in values):
            raise ValueError("Укажите короткие названия навыков через запятую.")
        return sorted({normalize_skill(value) for value in values})


class VacancyInput(Input):
    title: str = Field(min_length=3, max_length=180)
    company: str = Field(min_length=1, max_length=180)
    city: str = Field(default="Казахстан", min_length=1, max_length=100)
    requirements: str = Field(min_length=5, max_length=12000)
    description: str = Field(default="", max_length=12000)
    experience: str = Field(default="", max_length=180)
    salary: str = Field(default="", max_length=180)


class ShortlistInput(Input):
    vacancy_id: str = Field(min_length=1, max_length=100)


class MatchInput(Input):
    candidate_id: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=30)
    city: str | None = Field(default=None, max_length=100)


class ClarificationInput(Input):
    question: str = Field(min_length=5, max_length=1000)
    answer: str = Field(min_length=3, max_length=5000)


class ReviewInput(ShortlistInput):
    candidate_id: str = Field(min_length=1, max_length=100)
    status: Literal["new", "shortlisted", "hold"]


class FeedbackInput(Input):
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=4000)
    candidate_id: str | None = Field(default=None, max_length=100)


def identity(prefix):
    return prefix + uuid.uuid4().hex


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def create_app(database_path=None):
    app = FastAPI(title="Auto-HR API", version="1.0.0", description="Локальный MVP помощника HR для IT-вакансий Казахстана.")
    store = Store(database_path or os.getenv("AUTO_HR_DB") or ROOT / "backend/runtime/auto_hr.sqlite3")
    matching = Matching()
    app.state.store = store
    app.state.matching = matching

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        issue = exc.errors()[0]
        field = ".".join(str(part) for part in issue.get("loc", [])[1:])
        return JSONResponse(status_code=422, content={"detail": f"Проверьте поле «{field or 'запрос'}»: {issue.get('msg', 'некорректное значение')}."})

    @app.middleware("http")
    async def workspace_session(request: Request, call_next):
        # Bind local browser sessions to one browser profile. This is deliberately
        # not presented as employer authentication or multi-user production access.
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_FILE_BYTES + 512 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Размер запроса превышает 8 МБ. Загрузите файл меньшего размера."})
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin:
                source = urlparse(origin)
                target = urlparse(str(request.base_url))
                local = {"localhost", "127.0.0.1", "::1"}
                if source.netloc != target.netloc and not (source.hostname in local and target.hostname in local):
                    return JSONResponse(status_code=403, content={"detail": "Запрос с другого сайта запрещён. Откройте Auto-HR напрямую."})
        sid = request.cookies.get(COOKIE, "")
        new_session = not (re.fullmatch(r"[a-f0-9]{64}", sid) and store.session_exists(sid))
        if new_session:
            sid = uuid.uuid4().hex + uuid.uuid4().hex
            store.create_session(sid)
        request.state.sid = sid
        response = await call_next(request)
        if new_session:
            response.set_cookie(COOKIE, sid, httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=30 * 24 * 3600)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def candidate_for(request, candidate_id):
        record = store.get("candidates", request.state.sid, candidate_id)
        if record is None:
            raise HTTPException(404, "Кандидат не найден в этом рабочем пространстве.")
        return record

    def vacancy_for(request, vacancy_id):
        record = matching.by_id.get(vacancy_id) or store.get("vacancies", request.state.sid, vacancy_id)
        if record is None:
            raise HTTPException(404, "Вакансия не найдена в этом рабочем пространстве.")
        return record

    def add_candidate(request, name, text, filename="", is_demo=False):
        if len(store.all("candidates", request.state.sid)) >= 100:
            raise HTTPException(400, "В MVP доступно до 100 кандидатов. Удалите ненужные резюме.")
        try:
            text = validate_text(text)
        except ResumeError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Contact masking happens before persistence. Raw uploaded bytes are never saved.
        text = anonymize_text(text)
        if len(re.findall(r"[^\W\d_]", professional_text(text), re.UNICODE)) < 12:
            raise HTTPException(422, "В резюме нет достаточно профессиональной информации. Добавьте опыт работы, проекты или IT-навыки.")
        record = refresh_skills({"id": identity("c_"), "name": name, "filename": filename,
                                 "text": text, "notes": "", "is_demo": is_demo,
                                 "created_at": timestamp(), "clarifications": [],
                                 "manual_added": [], "manual_removed": [], "manual_confirmed": []})
        store.put("candidates", request.state.sid, record)
        return record

    def add_vacancy(request, data, is_demo=False):
        if len(store.all("vacancies", request.state.sid)) >= 100:
            raise HTTPException(400, "В MVP доступно до 100 собственных вакансий.")
        full = "\n".join((data.title, data.requirements, data.description))
        # Broad role vocabulary plus the shared technology dictionary permits
        # novel IT positions without admitting unrelated vacancies accidentally.
        it_role = re.search(r"\bIT\b|\bИТ\b|software|developer|programmer|backend|frontend|full.?stack|devops|mlops|\bSRE\b|\bQA\b|\bUX\b|\bUI\b|data\s+(?:scien|analyst|engineer)|product\s+(?:manager|owner)|machine\s+learning|(?:cloud|network|security|platform)\s+engineer|тестир|разработ|программ|информацион|кибер|системн\w*\s+(?:админ|инженер)|аналитик\s+данных|инженер.*(?:сет|данн|\bПО\b)|технич.*поддерж", full, re.I)
        if not it_role and not extract_skills(full):
            raise HTTPException(422, "Auto-HR MVP работает с IT-вакансиями. Укажите IT-роль и технические требования.")
        record = {**data.model_dump(), "id": identity("v_"), "skills": extract_skills(full),
                  "employment": "Полная занятость", "responsibilities": "", "url": "",
                  "is_custom": True, "is_demo": is_demo, "category": "IT", "publish_date": timestamp()[:10]}
        store.put("vacancies", request.state.sid, record)
        return record

    @app.get("/api/health")
    def health():
        return {"status": "ok", **matching.pipeline(), "vacancy_count": len(matching.catalogue), "catalogue_note": CATALOGUE_NOTE}

    @app.get("/api/vacancies")
    def vacancies(request: Request, q: str = Query(default="", max_length=200), city: str = Query(default="", max_length=100),
                  page: int = Query(default=1, ge=1), limit: int = Query(default=20, ge=1, le=100)):
        all_items = store.all("vacancies", request.state.sid) + matching.catalogue
        cities = sorted({item["city"] for item in all_items if item["city"]})
        items = [item for item in all_items if (not city or item["city"].casefold() == city.casefold())
                 and (not q or q.casefold() in " ".join((item["title"], item["company"], item["requirements"], item["description"], ", ".join(item["skills"]))).casefold())]
        return {"items": items[(page - 1) * limit:page * limit], "total": len(items), "page": page, "limit": limit, "cities": cities}

    @app.post("/api/vacancies", status_code=201)
    def create_vacancy(data: VacancyInput, request: Request):
        return add_vacancy(request, data)

    @app.get("/api/vacancies/{vacancy_id}")
    def get_vacancy(vacancy_id: str, request: Request):
        return vacancy_for(request, vacancy_id)

    @app.get("/api/candidates")
    def candidates(request: Request):
        return {"items": [public_candidate(row) for row in store.all("candidates", request.state.sid)]}

    @app.post("/api/candidates", status_code=201)
    def create_candidate(data: CandidateInput, request: Request):
        return public_candidate(add_candidate(request, data.name, data.text))

    @app.post("/api/candidates/upload", status_code=201)
    async def upload_candidate(request: Request, file: UploadFile = File(...), name: str = Form(default="")):
        if len(name.strip()) > 120:
            raise HTTPException(422, "Имя должно быть не длиннее 120 символов.")
        try:
            data = await file.read(MAX_FILE_BYTES + 1)
        finally:
            await file.close()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, "Размер файла превышает 8 МБ. Загрузите файл меньшего размера.")
        filename = Path(file.filename or "resume").name[:200]
        try:
            text = await run_in_threadpool(extract_resume, data, filename)
        except ResumeError as exc:
            raise HTTPException(422, str(exc)) from exc
        display_name = name.strip() or Path(filename).stem[:120] or "Кандидат"
        return public_candidate(add_candidate(request, display_name, text, filename=filename))

    @app.get("/api/candidates/{candidate_id}")
    def get_candidate(candidate_id: str, request: Request):
        return public_candidate(candidate_for(request, candidate_id))

    @app.patch("/api/candidates/{candidate_id}")
    def patch_candidate(candidate_id: str, data: CandidatePatch, request: Request):
        record = candidate_for(request, candidate_id)
        if data.name is not None:
            record["name"] = data.name
        if data.notes is not None:
            record["notes"] = data.notes
        if data.skills is not None:
            auto, desired = set(record["auto_skills"]), set(data.skills)
            record["manual_added"] = sorted(desired - auto)
            record["manual_removed"] = sorted(auto - desired)
            record["manual_confirmed"] = sorted(desired & auto)
        refresh_skills(record)
        store.put("candidates", request.state.sid, record)
        return public_candidate(record)

    @app.delete("/api/candidates/{candidate_id}")
    def delete_candidate(candidate_id: str, request: Request):
        candidate_for(request, candidate_id)
        store.delete_candidate(request.state.sid, candidate_id)
        return {"deleted": True}

    @app.post("/api/candidates/{candidate_id}/clarifications")
    def clarify(candidate_id: str, data: ClarificationInput, request: Request):
        record = candidate_for(request, candidate_id)
        if len(record["clarifications"]) >= 30:
            raise HTTPException(422, "Достигнут лимит: до 30 уточнений на кандидата.")
        answer = anonymize_text(data.answer)
        if len(record["text"]) + sum(len(row["answer"]) for row in record["clarifications"]) + len(answer) > MAX_TEXT_CHARS:
            raise HTTPException(422, "Суммарный текст резюме и ответов превышает 60 000 символов.")
        record["clarifications"].append({"question": data.question, "answer": answer, "created_at": timestamp(), "source": "self_reported"})
        refresh_skills(record)
        store.put("candidates", request.state.sid, record)
        return public_candidate(record)

    @app.post("/api/shortlist")
    def shortlist(data: ShortlistInput, request: Request):
        start = perf_counter()
        vacancy = vacancy_for(request, data.vacancy_id)
        rows = matching.shortlist(store.all("candidates", request.state.sid), vacancy, store.reviews(request.state.sid, data.vacancy_id))
        return {"vacancy": vacancy, "candidates": rows, "pipeline": matching.pipeline(), "elapsed_ms": round((perf_counter() - start) * 1000, 1)}

    @app.put("/api/reviews")
    def review(data: ReviewInput, request: Request):
        vacancy_for(request, data.vacancy_id)
        candidate_for(request, data.candidate_id)
        store.review(request.state.sid, data.vacancy_id, data.candidate_id, data.status)
        return {"status": data.status}

    @app.post("/api/matches")
    def matches(data: MatchInput, request: Request):
        record = candidate_for(request, data.candidate_id)
        rows = store.all("vacancies", request.state.sid) + matching.catalogue
        if data.city:
            rows = [row for row in rows if row["city"].casefold() == data.city.casefold()]
        items, pipeline = matching.matches(record, rows, data.limit)
        return {"items": items, "pipeline": pipeline}

    @app.get("/api/export")
    def export(request: Request, vacancy_id: str):
        vacancy = vacancy_for(request, vacancy_id)
        rows = matching.shortlist(store.all("candidates", request.state.sid), vacancy, store.reviews(request.state.sid, vacancy_id))
        stream = StringIO(newline="")
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Кандидат", "Вакансия", "Относительный балл ML2", "Покрытие навыков", "Подтверждены", "Не подтверждены", "Требуют уточнения", "Решение HR", "Объяснение", "Заметки HR"])
        statuses = {"new": "Не рассмотрен", "shortlisted": "В shortlist", "hold": "На рассмотрении"}
        def safe(value):
            if isinstance(value, (int, float)):
                return value
            text = str(value)
            return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text
        for row in rows:
            writer.writerow([safe(value) for value in [row["candidate"]["name"], vacancy["title"], row["score"],
                             f"{row['skill_score']:.0%}", ", ".join(row["matched_skills"]), ", ".join(row["missing_skills"]),
                             ", ".join(row["uncertain_skills"]), statuses[row["status"]], row["explanation"], row["candidate"]["notes"]]])
        return Response("\ufeff" + stream.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="auto-hr-shortlist.csv"'})

    @app.post("/api/feedback", status_code=201)
    def feedback(data: FeedbackInput, request: Request):
        if data.candidate_id:
            candidate_for(request, data.candidate_id)
        record = {"id": identity("f_"), **data.model_dump(), "created_at": timestamp()}
        store.put("feedback", request.state.sid, record)
        return {"id": record["id"]}

    @app.get("/api/stats")
    def stats(request: Request):
        reviews = store.reviews(request.state.sid)
        return {"candidate_count": len(store.all("candidates", request.state.sid)),
                "shortlisted_count": sum(row["status"] == "shortlisted" for row in reviews),
                "review_count": sum(row["status"] != "new" for row in reviews),
                "feedback_count": len(store.all("feedback", request.state.sid)),
                "custom_vacancy_count": len(store.all("vacancies", request.state.sid))}

    @app.post("/api/demo")
    def demo(request: Request):
        existing = next((row for row in store.all("vacancies", request.state.sid) if row.get("is_demo")), None)
        if existing:
            demo_candidates = [public_candidate(row) for row in store.all("candidates", request.state.sid) if row["is_demo"]]
            return {"vacancy_id": existing["id"], "candidates": demo_candidates}
        vacancy = add_vacancy(request, VacancyInput(title="Middle Python Backend Developer", company="Auto-HR · демо-команда",
                              city="Алматы", requirements="Обязательны Python, FastAPI, PostgreSQL, Docker, Git, REST API. Опыт backend разработки от 2 лет. Желателен Kubernetes.",
                              description="Разрабатывать backend сервисы и REST API, писать тесты, оптимизировать SQL и разворачивать сервисы.",
                              experience="От 2 лет", salary="Обсуждается на интервью"), is_demo=True)
        examples = [
            ("Кандидат А · backend", "Middle Python backend developer. Work experience: 4 years. Разрабатывал REST API на FastAPI и Python. PostgreSQL, SQL, Docker, Git. Писал unit tests и разворачивал сервисы в Kubernetes. В команде отвечал за backend сервис платежей, оптимизацию запросов и deployment."),
            ("Кандидат Б · fullstack", "Middle fullstack developer. Work experience: 3 years. Python, Django, PostgreSQL, Git, Docker, JavaScript, React, REST API. Разрабатывал клиентские интерфейсы и backend интернет-магазина. Знакомство с FastAPI, хочу изучить Kubernetes."),
            ("Кандидат В · начинающий", "Junior Python backend developer. Work experience: 1 year. Разработал учебный REST API с FastAPI и SQLite. Python, Git, SQL, unit tests. Docker изучаю на учебных проектах. В PostgreSQL и Kubernetes не работал."),
            ("Кандидат Г · frontend", "Middle frontend developer. Work experience: 3 years. React, TypeScript, JavaScript, HTML, CSS, Git, Figma, Jest. Разрабатывал интерфейсы личного кабинета и компоненты дизайн-системы. Работал с REST API со стороны клиента."),
        ]
        records = [public_candidate(add_candidate(request, name, text, is_demo=True)) for name, text in examples]
        return {"vacancy_id": vacancy["id"], "candidates": records}

    @app.delete("/api/session")
    def clear_session(request: Request):
        store.clear_session(request.state.sid)
        return {"deleted": True}

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "API-маршрут не найден.")
        dist = ROOT / "frontend/dist"
        candidate = (dist / path).resolve()
        if dist.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        index = dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(404, "Интерфейс ещё не собран. Выполните npm run build в папке frontend.")

    return app


app = create_app()
