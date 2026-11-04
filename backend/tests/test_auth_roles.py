"""Real auth, role boundaries, persistent accounts and legacy-data adoption."""

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.auth import AUTH_COOKIE, token_hash
from backend.matching import refresh_skills
from backend.tests.test_api import candidate, login, vacancy


@pytest.fixture
def app(tmp_path):
    return create_app(database_path=tmp_path / "roles.sqlite3")


def legacy_data(store, sid="a" * 64, suffix="one"):
    store.create_session(sid)
    record = refresh_skills({"id": "legacy-c-" + suffix, "name": "Сохранённое резюме", "filename": "resume.txt",
                            "text": "Python backend developer. Work experience: 3 years. Built FastAPI services.",
                            "notes": "Keep previous HR notes", "is_demo": False, "created_at": "2026-11-03",
                            "clarifications": [], "manual_added": [], "manual_removed": [], "manual_confirmed": []})
    vac = {"id": "legacy-v-" + suffix, "title": "Python Backend Developer", "company": "Legacy Company",
           "city": "Алматы", "requirements": "Python FastAPI", "description": "Backend services", "experience": "От 2 лет",
           "salary": "", "skills": ["python", "fastapi"], "url": "", "responsibilities": "", "is_custom": True}
    store.put("vacancies", sid, vac)
    store.put("candidates", sid, record, vacancy_ids=[vac["id"]])
    store.review(sid, vac["id"], record["id"], "shortlisted")
    store.put("feedback", sid, {"id": "legacy-f-" + suffix, "candidate_id": record["id"], "rating": 4})
    store.put("questions", sid, {"id": "legacy-q-" + suffix, "candidate_id": record["id"], "vacancy_id": vac["id"], "question": "Почему балл?"})
    return record, vac


def test_public_login_state_and_private_routes_require_auth(app, tmp_path, monkeypatch):
    # Test the public shell independently of an optional local frontend build.
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Public login shell</title>", encoding="utf-8")
    monkeypatch.setattr("backend.app.ROOT", tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/auth/me").json() == {"user": None}
        assert client.get("/api/health").status_code == 200
        shell = client.get("/")
        assert shell.status_code == 200
        assert "Public login shell" in shell.text
        for method, path in (("get", "/api/candidates"), ("get", "/api/vacancies"),
                             ("get", "/api/questions"), ("get", "/api/stats"),
                             ("post", "/api/candidates"), ("post", "/api/vacancies"),
                             ("post", "/api/questions"), ("post", "/api/matches"),
                             ("delete", "/api/session")):
            response = getattr(client, method)(path)
            assert response.status_code == 401, (method, path, response.text)
            assert response.headers["cache-control"] == "no-store"


def test_credentials_do_not_allow_client_selected_roles_or_forged_cookies(app):
    with TestClient(app) as client:
        assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
        assert client.post("/api/auth/login", json={"username": "missing", "password": "admin"}).status_code == 401
        assert client.post("/api/auth/login", json={"username": "user", "password": "user", "role": "hr"}).status_code == 422
        client.cookies.set(AUTH_COOKIE, "f" * 64)
        client.cookies.set("autohr_role", "hr")
        client.cookies.set("autohr_session", "a" * 64)
        assert client.get("/api/candidates").status_code == 401
        user = login(client, "user")
        assert user == {"username": "user", "role": "candidate", "display_name": "Кандидат"}
        assert set(client.get("/api/auth/me").json()["user"]) == {"username", "role", "display_name"}


def test_passwords_and_tokens_are_hashed_and_cookie_is_http_only(app):
    with TestClient(app) as client:
        response = client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        token = client.cookies.get(AUTH_COOKIE)
        assert len(token) == 64
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=strict" in response.headers["set-cookie"]
        assert token not in response.text
        with app.state.store.connect() as db:
            accounts = db.execute("SELECT username,password_hash FROM accounts").fetchall()
            assert all(row["password_hash"] != row["username"] and row["password_hash"].startswith("pbkdf2_sha256$") for row in accounts)
            tokens = db.execute("SELECT token_hash FROM auth_sessions").fetchall()
            assert tokens[0][0] == token_hash(token) and tokens[0][0] != token


def test_login_rotates_token_logout_revokes_it_and_expiry_is_server_enforced(app):
    with TestClient(app) as client, TestClient(app) as replay:
        login(client)
        old = client.cookies.get(AUTH_COOKIE)
        login(client)
        current = client.cookies.get(AUTH_COOKIE)
        assert old != current
        replay.cookies.set(AUTH_COOKIE, old)
        assert replay.get("/api/candidates").status_code == 401
        replay.cookies.set(AUTH_COOKIE, current)
        assert replay.get("/api/candidates").status_code == 200
        assert client.post("/api/auth/logout").json() == {"logged_out": True}
        assert replay.get("/api/candidates").status_code == 401
        login(client)
        with app.state.store.connect() as db:
            db.execute("UPDATE auth_sessions SET expires_at=0")
        assert client.get("/api/candidates").status_code == 401
        assert client.get("/api/auth/me").json()["user"] is None


def test_same_account_shares_persistent_workspace_across_logins_and_restart(app):
    with TestClient(app) as first, TestClient(app) as second:
        login(first)
        record = candidate(first)
        login(second)
        assert second.get(f"/api/candidates/{record['id']}").status_code == 200
        first.post("/api/auth/logout")
        assert second.get(f"/api/candidates/{record['id']}").status_code == 200
    with TestClient(create_app(database_path=app.state.store.path)) as reopened:
        login(reopened)
        assert reopened.get(f"/api/candidates/{record['id']}").status_code == 200


@pytest.mark.parametrize("method,path", [
    ("post", "/api/vacancies"), ("post", "/api/vacancies/import-preview"),
    ("delete", "/api/vacancies/11"), ("post", "/api/shortlist"),
    ("put", "/api/reviews"), ("get", "/api/export"),
    ("post", "/api/demo"), ("put", "/api/candidates/missing/vacancies"),
])
def test_candidate_cannot_call_hr_only_routes_even_with_forged_role(app, method, path):
    with TestClient(app) as client:
        login(client, "user")
        response = getattr(client, method)(path, headers={"X-Role": "hr"})
        assert response.status_code == 403, response.text


def test_shared_vacancies_and_private_resumes_are_separate(app):
    with TestClient(app) as hr, TestClient(app) as user:
        login(hr)
        target = vacancy(hr, company="Visible HR vacancy")
        private_hr_cv = candidate(hr)
        login(user, "user")
        assert user.get(f"/api/vacancies/{target['id']}").status_code == 200
        assert user.get("/api/vacancies", params={"q": "Visible HR vacancy"}).json()["items"][0]["id"] == target["id"]
        mine = candidate(user)
        assert user.get(f"/api/candidates/{private_hr_cv['id']}").status_code == 404
        assert hr.get(f"/api/candidates/{mine['id']}").status_code == 404
        assert user.patch(f"/api/candidates/{mine['id']}", json={"skills": ["docker"]}).status_code == 403
        assert user.patch(f"/api/candidates/{mine['id']}", json={"notes": "Pretend HR approval"}).status_code == 403
        assert user.patch(f"/api/candidates/{mine['id']}", json={"name": "Моё резюме"}).status_code == 200
        refused = user.post("/api/candidates", json={"name": "No automatic applications", "text": mine["text"], "vacancy_id": target["id"]})
        assert refused.status_code == 403
        assert len(user.get("/api/candidates").json()["items"]) == 1
        assert user.post("/api/matches", json={"candidate_id": mine["id"], "limit": 2}).status_code == 200
        assert user.post(f"/api/candidates/{mine['id']}/feedback", json={"vacancy_id": target["id"]}).status_code == 200
        assert user.delete(f"/api/candidates/{mine['id']}").status_code == 200
        assert hr.get(f"/api/candidates/{private_hr_cv['id']}").status_code == 200


def test_questions_have_real_scope_provenance_and_do_not_change_model_input(app):
    with TestClient(app) as hr, TestClient(app) as user:
        login(hr)
        target = vacancy(hr)
        private_hr = candidate(hr)
        login(user, "user")
        mine = candidate(user)
        original = user.get(f"/api/candidates/{mine['id']}").json()
        result = user.post("/api/questions", json={"candidate_id": mine["id"], "vacancy_id": target["id"], "question": "Почему у меня такой балл?"})
        assert result.status_code == 201, result.text
        answer = result.json()
        assert answer["answer"] and answer["evidence"] and answer.get("score_breakdown")
        assert user.get(f"/api/candidates/{mine['id']}").json() == original
        assert user.get("/api/questions").json()["items"][0]["id"] == answer["id"]
        assert hr.get("/api/questions").json()["items"] == []
        assert user.post("/api/questions", json={"candidate_id": private_hr["id"], "vacancy_id": target["id"], "question": "Почему балл?"}).status_code == 404
        assert user.get("/api/questions", params={"candidate_id": private_hr["id"]}).status_code == 404
        assert user.post(f"/api/candidates/{private_hr['id']}/feedback", json={"vacancy_id": target["id"]}).status_code == 404
        assert user.post("/api/questions", json={"vacancy_id": target["id"], "question": "a" * 1001}).status_code == 422
        user.delete(f"/api/candidates/{mine['id']}")
        assert user.get("/api/questions").json()["items"] == []
        question = user.post("/api/questions", json={"vacancy_id": target["id"], "question": "Какая зарплата?"}).json()
        assert question["id"]
        hr.delete(f"/api/vacancies/{target['id']}")
        assert user.get("/api/questions").json()["items"] == []


def test_feedback_without_vacancy_and_import_preview_do_not_publish_implicitly(app):
    with TestClient(app) as hr:
        login(hr)
        cv = candidate(hr)
        feedback = hr.post(f"/api/candidates/{cv['id']}/feedback", json={})
        assert feedback.status_code == 200 and feedback.json()["summary"]
        before = hr.get("/api/stats").json()["custom_vacancy_count"]
        text = "Должность: Python Backend Developer\nКомпания: Import Team\nГород: Алматы\nТребования: Python FastAPI Docker\nОпыт: От 2 лет"
        preview = hr.post("/api/vacancies/import-preview", files={"file": ("job.txt", text.encode(), "text/plain")})
        assert preview.status_code == 200, preview.text
        assert preview.json()["fields"]["company"] == "Import Team"
        assert hr.get("/api/stats").json()["custom_vacancy_count"] == before
        created = hr.post("/api/vacancies", json=preview.json()["fields"])
        assert created.status_code == 201, created.text
        bad = hr.post("/api/vacancies/import-preview", files={"file": ("job.exe", b"invalid", "application/octet-stream")})
        assert bad.status_code == 422


def test_clearing_hr_workspace_removes_published_vacancy_questions_only(app):
    with TestClient(app) as hr, TestClient(app) as user:
        login(hr)
        target = vacancy(hr)
        login(user, "user")
        mine = candidate(user)
        question = user.post("/api/questions", json={"candidate_id": mine["id"], "vacancy_id": target["id"], "question": "Какие требования?"})
        assert question.status_code == 201
        assert hr.delete("/api/session").status_code == 200
        assert user.get("/api/questions").json()["items"] == []
        assert user.get(f"/api/candidates/{mine['id']}").status_code == 200
        assert user.get(f"/api/vacancies/{target['id']}").status_code == 404


def test_created_vacancy_never_invents_full_time_employment(app):
    with TestClient(app) as hr, TestClient(app) as user:
        login(hr)
        target = vacancy(hr, description="Частичная занятость, 20 часов в неделю. Python backend API.")
        assert target["employment"] == ""
        assert "Частичная занятость" in target["description"]
        login(user, "user")
        saved = user.get(f"/api/vacancies/{target['id']}").json()
        assert saved["employment"] == ""
        answer = user.post("/api/questions", json={"vacancy_id": target["id"], "question": "Какая занятость у вакансии?"})
        assert answer.status_code == 201
        assert "полная занятость" not in answer.json()["answer"].casefold()


def test_candidate_first_keeps_legacy_cookie_then_hr_claims_only_that_workspace(app):
    record, target = legacy_data(app.state.store)
    other, _ = legacy_data(app.state.store, "b" * 64, "other")
    with TestClient(app) as client:
        client.cookies.set("autohr_session", "a" * 64)
        login(client, "user")
        assert client.cookies.get("autohr_session") == "a" * 64
        assert client.get("/api/candidates").json()["items"] == []
        client.post("/api/auth/logout")
        login(client)
        restored = client.get(f"/api/candidates/{record['id']}").json()
        assert restored["notes"] == record["notes"] and restored["vacancy_ids"] == [target["id"]]
        assert client.get("/api/stats").json()["shortlisted_count"] == 1
        assert client.get("/api/stats").json()["feedback_count"] == 1
        assert client.get("/api/questions").json()["items"][0]["candidate_id"] == record["id"]
        assert client.get(f"/api/candidates/{other['id']}").status_code == 404
        client.post("/api/auth/logout")
        client.cookies.set("autohr_session", "b" * 64)
        login(client)
        assert client.get(f"/api/candidates/{other['id']}").status_code == 404


def test_later_legacy_claim_preserves_new_hr_data_and_cannot_steal_account_workspace(app):
    legacy, _ = legacy_data(app.state.store)
    with TestClient(app) as hr, TestClient(app) as user:
        login(hr)  # No legacy cookie must not consume the one eligible adoption.
        new = candidate(hr)
        hr.post("/api/auth/logout")
        login(user, "user")
        private = candidate(user)
        candidate_sid = app.state.store.authenticate(user.cookies.get(AUTH_COOKIE))["workspace_id"]
        hr.cookies.set("autohr_session", candidate_sid)
        login(hr)
        assert hr.get(f"/api/candidates/{private['id']}").status_code == 404
        hr.post("/api/auth/logout")
        hr.cookies.set("autohr_session", "a" * 64)
        login(hr)
        assert hr.get(f"/api/candidates/{legacy['id']}").status_code == 200
        assert hr.get(f"/api/candidates/{new['id']}").status_code == 200
        assert user.get(f"/api/candidates/{private['id']}").status_code == 200
