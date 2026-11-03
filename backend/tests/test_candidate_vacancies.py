"""Regression coverage for explicit applications and deletion boundaries."""
import csv
import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.tests.test_api import candidate, login, shortlist, vacancy


@pytest.fixture
def app(tmp_path):
    return create_app(database_path=tmp_path / "applications.sqlite3")


@pytest.fixture
def client(app):
    with TestClient(app) as current:
        login(current)
        yield current


def assign(client, candidate_id, *vacancy_ids):
    response = client.put(f"/api/candidates/{candidate_id}/vacancies", json={"vacancy_ids": list(vacancy_ids)})
    assert response.status_code == 200, response.text
    return response.json()


def ranked_ids(client, vacancy_id):
    return {row["candidate"]["id"] for row in shortlist(client, vacancy_id)["candidates"]}


def csv_rows(client, vacancy_id):
    response = client.get("/api/export", params={"vacancy_id": vacancy_id})
    assert response.status_code == 200, response.text
    return list(csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"))


def test_upload_is_only_attached_to_the_chosen_vacancy(client):
    one, two = vacancy(client), vacancy(client, title="Frontend React Developer", requirements="React, TypeScript, CSS")
    text = candidate(client, name="Explicit JSON", vacancy_id=one["id"])
    uploaded = client.post("/api/candidates/upload", data={"vacancy_id": two["id"]},
                           files={"file": ("frontend.txt", b"Middle frontend developer. Work experience: 3 years. React TypeScript and CSS.", "text/plain")})
    assert uploaded.status_code == 201, uploaded.text
    pool = candidate(client, name="General talent pool")
    assert text["vacancy_ids"] == [one["id"]]
    assert uploaded.json()["vacancy_ids"] == [two["id"]]
    assert pool["vacancy_ids"] == []
    assert ranked_ids(client, one["id"]) == {text["id"]}
    assert ranked_ids(client, two["id"]) == {uploaded.json()["id"]}
    assert len(client.get("/api/candidates").json()["items"]) == 3
    assert {row["id"] for row in client.get("/api/candidates", params={"vacancy_id": one["id"]}).json()["items"]} == {text["id"]}
    assert client.get("/api/candidates", params={"vacancy_id": "missing"}).status_code == 404
    # Candidate-to-job discovery is allowed but never creates applications.
    assert client.post("/api/matches", json={"candidate_id": pool["id"], "limit": 2}).status_code == 200
    assert client.get(f"/api/candidates/{pool['id']}").json()["vacancy_ids"] == []
    assert ranked_ids(client, one["id"]) == {text["id"]}


def test_empty_vacancy_does_not_borrow_candidates_or_export_from_other_vacancies(client):
    one, two = vacancy(client), vacancy(client)
    first = candidate(client, name="Only in vacancy one", vacancy_id=one["id"])
    second = candidate(client, name="Only in pool")
    assert ranked_ids(client, two["id"]) == set()
    assert len(csv_rows(client, two["id"])) == 1  # Only the column headings.
    assert first["name"] in csv_rows(client, one["id"])[1]
    assert second["name"] not in {cell for row in csv_rows(client, one["id"]) for cell in row}
    refused = client.put("/api/reviews", json={"vacancy_id": two["id"], "candidate_id": first["id"], "status": "shortlisted"})
    assert refused.status_code == 409
    assert client.get("/api/stats").json()["shortlisted_count"] == 0


def test_link_unlink_preserves_other_vacancy_and_its_hr_decision(client):
    one, two = vacancy(client), vacancy(client)
    person = candidate(client, vacancy_id=one["id"])
    linked = assign(client, person["id"], one["id"], two["id"], one["id"])
    assert linked["vacancy_ids"] == sorted([one["id"], two["id"]])
    for target, status in ((one, "shortlisted"), (two, "hold")):
        response = client.put("/api/reviews", json={"vacancy_id": target["id"], "candidate_id": person["id"], "status": status})
        assert response.status_code == 200
    assign(client, person["id"], one["id"])
    assert ranked_ids(client, two["id"]) == set()
    assert shortlist(client, one["id"])["candidates"][0]["status"] == "shortlisted"
    assert client.get("/api/stats").json()["review_count"] == 1
    assign(client, person["id"], one["id"], two["id"])
    assert shortlist(client, two["id"])["candidates"][0]["status"] == "new"
    assign(client, person["id"])
    assert client.get(f"/api/candidates/{person['id']}").status_code == 200
    assert client.get("/api/stats").json()["review_count"] == 0


def test_invalid_or_foreign_vacancy_does_not_create_or_partially_reassign_cv(app, client):
    own = vacancy(client)
    person = candidate(client, vacancy_id=own["id"])
    client.put("/api/reviews", json={"vacancy_id": own["id"], "candidate_id": person["id"], "status": "shortlisted"})
    orphan_sid = "f" * 64
    app.state.store.create_session(orphan_sid)
    foreign = {**own, "id": "unpublished-legacy-vacancy"}
    app.state.store.put("vacancies", orphan_sid, foreign)
    with TestClient(app) as stranger:
        login(stranger, "user")
        for invalid in ("missing-vacancy", foreign["id"]):
            response = client.put(f"/api/candidates/{person['id']}/vacancies", json={"vacancy_ids": [invalid]})
            assert response.status_code == 404
            response = client.put(f"/api/candidates/{person['id']}/vacancies", json={"vacancy_ids": [own["id"], invalid]})
            assert response.status_code == 404
            response = client.post("/api/candidates", json={"name": "Must not persist", "text": "Python backend developer with Docker and SQL experience.", "vacancy_id": invalid})
            assert response.status_code == 404
            response = client.post("/api/candidates/upload", data={"vacancy_id": invalid},
                                   files={"file": ("resume.txt", b"Python backend developer with Docker and SQL experience.", "text/plain")})
            assert response.status_code == 404
        assert stranger.put(f"/api/candidates/{person['id']}/vacancies", json={"vacancy_ids": [foreign["id"]]}).status_code == 403
        assert stranger.get("/api/candidates", params={"vacancy_id": own["id"]}).json()["items"] == []
        assert stranger.delete(f"/api/vacancies/{own['id']}").status_code == 403
    stored = client.get(f"/api/candidates/{person['id']}").json()
    assert stored["vacancy_ids"] == [own["id"]]
    assert len(client.get("/api/candidates").json()["items"]) == 1
    assert shortlist(client, own["id"])["candidates"][0]["status"] == "shortlisted"


def test_deleting_vacancy_removes_only_its_links_and_decisions(client, app):
    one, two = vacancy(client), vacancy(client)
    person = candidate(client, vacancy_id=one["id"])
    assign(client, person["id"], one["id"], two["id"])
    for target in (one, two):
        assert client.put("/api/reviews", json={"vacancy_id": target["id"], "candidate_id": person["id"], "status": "shortlisted"}).status_code == 200
    assert client.delete(f"/api/vacancies/{one['id']}").status_code == 200
    assert client.delete(f"/api/vacancies/{one['id']}").status_code == 404
    assert client.get(f"/api/vacancies/{one['id']}").status_code == 404
    assert client.get(f"/api/candidates/{person['id']}").json()["vacancy_ids"] == [two["id"]]
    assert shortlist(client, two["id"])["candidates"][0]["status"] == "shortlisted"
    assert client.get("/api/stats").json()["shortlisted_count"] == 1
    assert client.get("/api/stats").json()["custom_vacancy_count"] == 1
    cookies = dict(client.cookies)
    with TestClient(create_app(database_path=app.state.store.path)) as reopened:
        reopened.cookies.update(cookies)
        assert reopened.get(f"/api/candidates/{person['id']}").json()["vacancy_ids"] == [two["id"]]
        assert reopened.get(f"/api/vacancies/{one['id']}").status_code == 404


def test_deleting_cv_cascades_all_applications_and_feedback(client, app):
    one, two = vacancy(client), vacancy(client)
    person = candidate(client, vacancy_id=one["id"])
    assign(client, person["id"], one["id"], two["id"])
    client.put("/api/reviews", json={"vacancy_id": one["id"], "candidate_id": person["id"], "status": "shortlisted"})
    client.post("/api/feedback", json={"rating": 4, "candidate_id": person["id"]})
    assert client.delete(f"/api/candidates/{person['id']}").status_code == 200
    assert client.delete(f"/api/candidates/{person['id']}").status_code == 404
    assert ranked_ids(client, one["id"]) == ranked_ids(client, two["id"]) == set()
    stats = client.get("/api/stats").json()
    assert stats["candidate_count"] == stats["shortlisted_count"] == stats["feedback_count"] == 0
    with app.state.store.connect() as db:
        assert db.execute("SELECT count(*) FROM candidate_vacancies").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0


def test_archive_cannot_be_deleted_but_can_have_explicit_candidates(client):
    archive = client.get("/api/vacancies", params={"limit": 1}).json()["items"][0]
    person = candidate(client, vacancy_id=archive["id"])
    assert client.delete(f"/api/vacancies/{archive['id']}").status_code == 403
    assert client.get(f"/api/vacancies/{archive['id']}").status_code == 200
    assert ranked_ids(client, archive["id"]) == {person["id"]}
    assert client.get("/api/health").json()["vacancy_count"] == 1017


def test_demo_links_only_demo_candidates_and_is_idempotent(client):
    unrelated = candidate(client)
    first = client.post("/api/demo").json()
    assert len(first["candidates"]) == 4
    assert all(person["vacancy_ids"] == [first["vacancy_id"]] for person in first["candidates"])
    assert ranked_ids(client, first["vacancy_id"]) == {person["id"] for person in first["candidates"]}
    assert unrelated["id"] not in ranked_ids(client, first["vacancy_id"])
    repeated = client.post("/api/demo").json()
    assert {person["id"] for person in repeated["candidates"]} == {person["id"] for person in first["candidates"]}
    assert len(client.get("/api/candidates").json()["items"]) == 5


def test_recreating_deleted_demo_vacancy_reuses_its_surviving_resumes(client):
    first = client.post("/api/demo").json()
    original_ids = {row["id"] for row in first["candidates"]}
    assert client.delete(f"/api/vacancies/{first['vacancy_id']}").status_code == 200
    assert all(not row["vacancy_ids"] for row in client.get("/api/candidates").json()["items"])
    recreated = client.post("/api/demo").json()
    assert recreated["vacancy_id"] != first["vacancy_id"]
    assert {row["id"] for row in recreated["candidates"]} == original_ids
    repeated = client.post("/api/demo").json()
    assert {row["id"] for row in repeated["candidates"]} == original_ids
    assert len(client.get("/api/candidates").json()["items"]) == 4


def test_unlink_survives_restart_and_session_clear_deletes_links(client, app):
    target = vacancy(client)
    person = candidate(client, vacancy_id=target["id"])
    client.put("/api/reviews", json={"vacancy_id": target["id"], "candidate_id": person["id"], "status": "hold"})
    assign(client, person["id"])
    cookies = dict(client.cookies)
    with TestClient(create_app(database_path=app.state.store.path)) as reopened:
        reopened.cookies.update(cookies)
        assert ranked_ids(reopened, target["id"]) == set()
        assert reopened.get(f"/api/candidates/{person['id']}").json()["vacancy_ids"] == []
        assert reopened.get("/api/stats").json()["review_count"] == 0
        assign(reopened, person["id"], target["id"])
        assert reopened.delete("/api/session").status_code == 200
    with app.state.store.connect() as db:
        assert db.execute("SELECT count(*) FROM candidate_vacancies").fetchone()[0] == 0


def test_legacy_database_migration_preserves_pool_and_only_valid_reviewed_pairs(tmp_path, client, app):
    source = candidate(client)
    sid, other_sid = "a" * 64, "b" * 64
    archive_id = next(iter(app.state.matching.by_id))
    db_path = tmp_path / "legacy.sqlite3"
    # Construct the actual pre-association schema instead of pre-marking a new DB.
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE sessions(id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE candidates(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE vacancies(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE reviews(session_id TEXT, vacancy_id TEXT, candidate_id TEXT, status TEXT,
                                 PRIMARY KEY(session_id,vacancy_id,candidate_id));
        """)
        db.executemany("INSERT INTO sessions(id) VALUES (?)", [(sid,), (other_sid,)])
        for cid, owner in (("reviewed", sid), ("pool", sid), ("foreign", other_sid)):
            record = {**source, "id": cid}
            db.execute("INSERT INTO candidates VALUES (?,?,?)", (cid, owner, json.dumps(record)))
        for vid, owner in (("own-vacancy", sid), ("foreign-vacancy", other_sid)):
            db.execute("INSERT INTO vacancies VALUES (?,?,?)", (vid, owner, json.dumps({"id": vid, "title": "Python Developer"})))
        db.executemany("INSERT INTO reviews VALUES (?,?,?,?)", [
            (sid, "own-vacancy", "reviewed", "shortlisted"),
            (sid, archive_id, "reviewed", "hold"),
            (sid, "missing-vacancy", "pool", "hold"),
            (sid, "foreign-vacancy", "pool", "hold"),
            (sid, "own-vacancy", "foreign", "hold"),
        ])
    migrated = create_app(database_path=db_path)
    with TestClient(migrated) as current:
        current.cookies.set("autohr_session", sid)
        login(current)
        records = {row["id"]: row for row in current.get("/api/candidates").json()["items"]}
        assert records["pool"]["vacancy_ids"] == []
        assert set(records["reviewed"]["vacancy_ids"]) == {"own-vacancy", archive_id}
        assert set(records) == {"reviewed", "pool"}
        assert current.get("/api/stats").json()["review_count"] == 2
    # Restarting is additive and idempotent; it does not re-import removed pairs.
    admin_sid = migrated.state.store.authenticate(current.cookies.get("autohr_auth"))["workspace_id"]
    migrated.state.store.set_candidate_vacancies(admin_sid, "reviewed", [])
    reopened = create_app(database_path=db_path)
    assert reopened.state.store.get("candidates", admin_sid, "reviewed")["vacancy_ids"] == []
    assert reopened.state.store.get("candidates", admin_sid, "pool")["vacancy_ids"] == []
