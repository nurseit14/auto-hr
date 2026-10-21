"""Convert actual model answers into editable Label Studio pre-annotations."""

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ML2.model import save_json

ROOT = Path(__file__).resolve().parent.parent


def make_task(pair: dict, record: dict | None, dataset_version: str, error: str = "") -> dict:
    cv, vacancy = pair["cv"], pair["vacancy"]
    answer = record["assessment"] if record else None
    warnings = list(answer["warnings"]) if answer else []
    if not cv["text"].strip():
        warnings.append("Резюме пустое: сведения о кандидате отсутствуют; требуется проверить исходный файл.")
    if error and not record:
        warnings.append("Предразметка не выполнена: " + error)
    data = {
        "pair_id": pair["pair_id"], "cv_id": cv["cv_id"], "vacancy_id": vacancy["vacancy_id"],
        "pair_caption": f"{cv['cv_id']} · вакансия {pair['selection_rank']} из 5 · {vacancy['title']} (ID {vacancy['vacancy_id']})",
        "cv_text": cv["text"], "vacancy_display": vacancy["display"],
        "vacancy_title": vacancy["title"], "vacancy_fields": vacancy,
        "cv_text_sha256": cv["text_sha256"], "vacancy_text_sha256": vacancy["source_sha256"],
        "dataset_version": dataset_version, "selection_rank": pair["selection_rank"],
        "selection_method": pair["selection_method"], "selection_score": pair["selection_score"],
        "preannotation_status": "ready" if record else "failed",
        "warnings_text": "\n".join(dict.fromkeys(warnings)) or "Предразметка модели: требуется ручная проверка.",
        "model_assessment": answer,
    }
    task = {"data": data, "predictions": []}
    if not record:
        return task
    results = []
    for index, row in enumerate(answer["requirements"], 1):
        start = vacancy["display"].index(row["vacancy_quote"])
        end = start + len(row["vacancy_quote"])
        region_id = f"r{index:03}"
        def result(name, kind, value):
            return {"id": region_id, "from_name": name, "to_name": "requirements", "type": kind,
                    "value": {"start": start, "end": end, **value}}
        results.append(result("requirement_type", "labels", {"text": row["vacancy_quote"], "labels": [row["category"]]}))
        results.append(result("requirement_title", "textarea", {"text": [row["requirement"]]}))
        results.append(result("match_score", "choices", {"choices": [str(row["score"])]}))
        results.append(result("reason", "textarea", {"text": [row["explanation"]]}))
        # Empty textarea predictions are invalid in some Label Studio versions.
        if row["cv_quote"]:
            results.append(result("cv_evidence", "textarea", {"text": [row["cv_quote"]]}))
    results.extend([
        {"id": "overall", "from_name": "overall_score", "to_name": "requirements", "type": "choices",
         "value": {"choices": [str(answer["overall_score"])]}},
        {"id": "overall_reason", "from_name": "overall_reason", "to_name": "requirements", "type": "textarea",
         "value": {"text": [answer["overall_explanation"]]}},
    ])
    task["predictions"] = [{"model_version": record["model_version"], "result": results}]
    return task


def local_token() -> str:
    """Use the existing local user's token; never create accounts or expose it."""
    token = os.getenv("LABEL_STUDIO_API_KEY")
    if token:
        return token
    # Respect modern Label Studio authentication: mint a short-lived access token
    # for the sole local user without enabling legacy auth or changing passwords.
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from label_studio.server import _setup_env
            _setup_env()
            from django.contrib.auth import get_user_model
            from rest_framework_simplejwt.tokens import AccessToken
            users = get_user_model().objects.filter(is_active=True)
            if users.count() != 1:
                raise ValueError("Multiple local users: set LABEL_STUDIO_API_KEY explicitly")
            user = users.get()
            if not user.active_organization.jwt.api_tokens_enabled:
                raise ValueError("JWT API tokens are disabled for the local organization")
            return str(AccessToken.for_user(user))
    except ImportError:
        pass
    candidates = [Path.home() / "Library/Application Support/label-studio/label_studio.sqlite3",
                  Path.home() / ".local/share/label-studio/label_studio.sqlite3"]
    if os.getenv("LABEL_STUDIO_BASE_DATA_DIR"):
        candidates.insert(0, Path(os.environ["LABEL_STUDIO_BASE_DATA_DIR"]) / "label_studio.sqlite3")
    for path in candidates:
        if path.exists():
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                rows = db.execute("SELECT key FROM authtoken_token ORDER BY created LIMIT 2").fetchall()
                if len(rows) == 1:
                    return rows[0][0]
    raise ValueError("Set LABEL_STUDIO_API_KEY; no unambiguous local token found")


def api(url: str, token: str, method: str, route: str, body=None):
    request = Request(url.rstrip("/") + route, method=method,
                      headers={"Authorization": ("Bearer " if token.count(".") == 2 else "Token ") + token,
                               "Content-Type": "application/json"},
                      data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"))
    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
            return json.loads(payload) if payload else None
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Label Studio API {method} {route}: HTTP {error.code}: {detail}") from None


def ensure_server(url: str, output: Path) -> None:
    parsed = urlparse(url)
    try:
        with urlopen(url.rstrip("/") + "/health", timeout=3):
            return
    except (URLError, OSError):
        pass
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Cannot start a remote Label Studio server")
    executable = Path(sys.executable).parent / "label-studio"
    if not executable.exists():
        raise ValueError("Run using the Python environment containing label-studio")
    with (output / "label_studio_server.log").open("ab") as log:
        subprocess.Popen([str(executable), "start", "--no-browser", "--internal-host", "127.0.0.1",
                          "--port", str(parsed.port or 8080)], cwd=ROOT,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urlopen(url.rstrip("/") + "/health", timeout=2):
                return
        except (URLError, OSError):
            time.sleep(1)
    raise RuntimeError("Label Studio did not start; inspect label_studio_server.log")


def publish(tasks: list[dict], output: Path, url: str, title: str, model_version: str) -> dict:
    if not tasks or any(not task.get("predictions") for task in tasks):
        raise ValueError("Finish all model predictions before publishing")
    output.mkdir(parents=True, exist_ok=True)
    ensure_server(url, output)
    is_local = urlparse(url).hostname in {"localhost", "127.0.0.1"}
    if not is_local and not os.getenv("LABEL_STUDIO_API_KEY"):
        raise ValueError("A remote server needs its own LABEL_STUDIO_API_KEY")
    token = local_token() if is_local else os.environ["LABEL_STUDIO_API_KEY"]
    state_file = output / "label_studio_project.json"
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["server_url"] != url.rstrip("/"):
            raise ValueError("This output folder is linked to a different Label Studio server")
        project_id = state["project_id"]
        project = api(url, token, "GET", f"/api/projects/{project_id}/")
        if project["model_version"] != model_version:
            raise ValueError("Project model version differs; use a new output folder")
    else:
        config = (Path(__file__).with_name("label_config.xml")).read_text(encoding="utf-8")
        project = api(url, token, "POST", "/api/projects/", {
            "title": title, "label_config": config, "show_collab_predictions": True,
            "model_version": model_version, "sampling": "Sequential sampling",
            "description": "ML2: по 5 вакансий на резюме. Баллы требований 0/1/2/3, объяснения, цитаты и общая оценка. Все ответы модели требуют проверки.",
            "show_instruction": True,
            "expert_instruction": "<p>Нажми на выделенное требование: проверь балл и объяснение. 0 — нет сведений; 1 — подтверждённое несоответствие; 2 — частичное соответствие; 3 — соответствие. Проверь общую оценку. Отметь подтверждение ручной проверки и нажми Submit.</p>",
        })
        project_id = project["id"]
        state = {"server_url": url.rstrip("/"), "project_id": project_id,
                 "project_url": f"{url.rstrip('/')}/projects/{project_id}/data", "title": title}
        save_json(state_file, state)
    # Only add missing pairs. Never modify or replace a human annotation.
    existing = {}
    existing_ids = {}
    page = 1
    while True:
        batch = api(url, token, "GET", f"/api/tasks/?project={project_id}&page={page}&page_size=100")
        rows = batch.get("tasks", batch.get("results", [])) if isinstance(batch, dict) else batch
        for row in rows:
            existing[row["data"].get("pair_id")] = row["data"]
            existing_ids[row["data"].get("pair_id")] = row["id"]
        if len(rows) < 100:
            break
        page += 1
    for task in tasks:
        old = existing.get(task["data"]["pair_id"])
        if old and (old.get("cv_text_sha256") != task["data"]["cv_text_sha256"] or
                    old.get("vacancy_text_sha256") != task["data"]["vacancy_text_sha256"]):
            raise ValueError("Existing pair has different source hashes; use a new output folder")
        if old and old.get("warnings_text") != task["data"]["warnings_text"]:
            # Refresh display-only metadata; source texts, predictions and human work stay intact.
            data = {**old, "warnings_text": task["data"]["warnings_text"]}
            api(url, token, "PATCH", f"/api/tasks/{existing_ids[task['data']['pair_id']]}/", {"data": data})
    missing = [t for t in tasks if t["data"]["pair_id"] not in existing]
    for start in range(0, len(missing), 40):
        api(url, token, "POST", f"/api/projects/{project_id}/import?commit_to_project=true", missing[start:start + 40])
    state["task_count"] = len(tasks)
    save_json(state_file, state)
    return state
