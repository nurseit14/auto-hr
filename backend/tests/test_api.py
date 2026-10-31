"""HTTP integration coverage for the local Auto-HR MVP.

Each fixture uses a real temporary SQLite database and the actual trained ML2
ranker; no model scores or persistence operations are mocked.
"""
import csv
import io
from pathlib import Path
import zipfile

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(database_path=tmp_path / "autohr.sqlite3")


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def vacancy(client, **updates):
    payload = {
        "title": "Python Backend Developer",
        "company": "Test Company",
        "city": "Алматы",
        "requirements": "Python, FastAPI, PostgreSQL, Docker. Work experience: 2 years.",
        "description": "Build and maintain backend REST API services.",
        "experience": "От 1 года до 3 лет",
        "salary": "500 000–800 000 ₸",
    }
    payload.update(updates)
    r = client.post("/api/vacancies", json=payload)
    assert r.status_code in (200, 201), r.text
    return r.json()


def candidate(client, **updates):
    payload = {
        "name": "Тестовый кандидат",
        "text": "Middle Python backend developer. Work experience: 3 years. Built FastAPI services with PostgreSQL and Docker. Implemented REST API and unit tests.",
    }
    payload.update(updates)
    r = client.post("/api/candidates", json=payload)
    assert r.status_code in (200, 201), r.text
    return r.json()


def shortlist(client, vid):
    r = client.post("/api/shortlist", json={"vacancy_id": vid})
    assert r.status_code == 200, r.text
    return r.json()


def test_real_catalogue_and_engine_are_reported(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    h = health.json()
    assert h["status"] == "ok"
    assert h["vacancy_count"] >= 1017
    assert h["ml2_available"] is True
    assert isinstance(h["ml1_available"], bool)
    assert h["engine"]
    assert h["catalogue_note"]
    catalog = client.get("/api/vacancies", params={"limit": 3}).json()
    assert catalog["total"] >= 1017
    assert len(catalog["items"]) == 3
    assert catalog["cities"]
    item = catalog["items"][0]
    assert item["id"] and item["title"]
    assert isinstance(item["skills"], list)
    assert client.get(f"/api/vacancies/{item['id']}").status_code == 200
    filtered = client.get("/api/vacancies", params={"q": "Python", "limit": 5}).json()
    assert filtered["total"] > 0
    assert len(filtered["items"]) <= 5


def test_session_isolation_and_private_ids(app):
    with TestClient(app) as one, TestClient(app) as two:
        v = vacancy(one)
        c = candidate(one)
        assert len(one.get("/api/candidates").json()["items"]) == 1
        assert two.get("/api/candidates").json()["items"] == []
        assert two.get(f"/api/vacancies/{v['id']}").status_code == 404
        assert two.patch(f"/api/candidates/{c['id']}", json={"notes": "No access"}).status_code == 404
        assert two.delete(f"/api/candidates/{c['id']}").status_code == 404
        assert two.post("/api/shortlist", json={"vacancy_id": v["id"]}).status_code == 404
        assert two.put("/api/reviews", json={"vacancy_id": v["id"], "candidate_id": c["id"], "status": "shortlisted"}).status_code == 404
        assert two.get("/api/export", params={"vacancy_id": v["id"]}).status_code == 404
        assert not any(x["id"] == v["id"] for x in two.get("/api/vacancies", params={"limit": 100}).json()["items"])
        assert one.cookies.get("autohr_session") or one.cookies


def test_shortlist_explanations_and_hr_review_persist(client):
    v = vacancy(client)
    good = candidate(client)
    weak = candidate(client, name="Дизайнер", text="UX UI designer. I build user research reports and prototypes in Figma. Work experience: 3 years.")
    data = shortlist(client, v["id"])
    assert data["vacancy"]["id"] == v["id"]
    assert data["pipeline"]["ml2_available"] is True
    assert data["elapsed_ms"] >= 0
    rows = data["candidates"]
    assert len(rows) == 2
    assert rows[0]["candidate"]["id"] == good["id"]
    assert rows[0]["score"] >= rows[1]["score"]
    assert all(isinstance(x["score"], (int, float)) for x in rows)
    for row in rows:
        assert 0 <= row["skill_score"] <= 1
        assert row["explanation"]
        assert set(row["matched_skills"]).isdisjoint(row["missing_skills"])
        assert row["status"] == "new"
        assert isinstance(row["questions"], list)
    assert "python" in rows[0]["matched_skills"]
    assert "python" in rows[1]["missing_skills"]
    assert any(x["skill"] == "python" and x["quote"] for x in rows[0]["evidence"])
    result = client.put("/api/reviews", json={"vacancy_id": v["id"], "candidate_id": weak["id"], "status": "shortlisted"})
    assert result.status_code == 200
    rows_after = shortlist(client, v["id"])["candidates"]
    assert next(x for x in rows_after if x["candidate"]["id"] == weak["id"])["status"] == "shortlisted"
    assert client.get("/api/stats").json()["shortlisted_count"] == 1


def test_manual_skill_corrections_and_clarification_change_evidence(client):
    v = vacancy(client, requirements="Python, Kubernetes, Docker")
    c = candidate(client, text="Backend developer with Python experience. Built production REST API services for two years.")
    before = shortlist(client, v["id"])["candidates"][0]
    assert "python" in before["matched_skills"]
    patched = client.patch(f"/api/candidates/{c['id']}", json={"skills": ["docker"], "notes": "HR verified the skill correction"})
    assert patched.status_code == 200, patched.text
    assert patched.json()["skills"] == ["docker"]
    corrected = shortlist(client, v["id"])["candidates"][0]
    assert "docker" in corrected["matched_skills"]
    assert "python" not in corrected["matched_skills"]
    assert "python" in corrected["missing_skills"]
    answer = client.post(f"/api/candidates/{c['id']}/clarifications", json={"question": "Какой у вас практический опыт с Kubernetes?", "answer": "В течение года разворачивал сервисы в Kubernetes, настраивал deployment и service."})
    assert answer.status_code in (200, 201), answer.text
    assert answer.json()["clarifications"][-1]["answer"].startswith("В течение года")
    after = shortlist(client, v["id"])["candidates"][0]
    assert "kubernetes" in after["matched_skills"]
    kube = next(x for x in after["evidence"] if x["skill"] == "kubernetes")
    assert "Kubernetes" in kube["quote"]
    assert kube["source"]
    assert after["candidate"]["notes"] == "HR verified the skill correction"


def test_candidate_anonymization_and_delete(client):
    c = candidate(client, text="Python backend developer with Docker and Git. Email: test@example.com. Телефон: +7 777 123 45 67.")
    assert "test@example.com" not in c["text"]
    assert "+7 777 123 45 67" not in c["text"]
    assert "python" in c["skills"]
    assert client.delete(f"/api/candidates/{c['id']}").status_code in (200, 204)
    assert client.get("/api/candidates").json()["items"] == []
    assert client.delete(f"/api/candidates/{c['id']}").status_code == 404


def test_candidates_persist_when_app_reopens(tmp_path):
    db = tmp_path / "persistent.sqlite3"
    with TestClient(create_app(database_path=db)) as first:
        c = candidate(first)
        cookies = dict(first.cookies)
    with TestClient(create_app(database_path=db)) as second:
        second.cookies.update(cookies)
        stored = second.get("/api/candidates").json()["items"]
        assert [x["id"] for x in stored] == [c["id"]]


def test_text_and_docx_upload(client):
    r = client.post("/api/candidates/upload", files={"file": ("resume.txt", "Python backend developer. Built FastAPI and Docker services with PostgreSQL.".encode(), "text/plain")})
    assert r.status_code in (200, 201), r.text
    assert r.json()["filename"] == "resume.txt"
    assert "python" in r.json()["skills"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        z.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Python developer. FastAPI PostgreSQL and Docker. Work experience: 3 years.</w:t></w:r></w:p></w:body></w:document>')
    doc = client.post("/api/candidates/upload", files={"file": ("resume.docx", stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert doc.status_code in (200, 201), doc.text
    assert "postgresql" in doc.json()["skills"]


@pytest.mark.parametrize("filename,data,mime", [
    ("empty.txt", b"   \n", "text/plain"),
    ("binary.txt", b"\xff\xfe\x00\x01\x02", "text/plain"),
    ("resume.exe", b"Not a resume", "application/octet-stream"),
    ("broken.pdf", b"%PDF-not-valid", "application/pdf"),
    ("broken.docx", b"not-a-zip", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
])
def test_bad_upload_has_actionable_json_error(client, filename, data, mime):
    r = client.post("/api/candidates/upload", files={"file": (filename, data, mime)})
    assert r.status_code in (400, 413, 415, 422), r.text
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]
    assert client.get("/api/candidates").json()["items"] == []


def test_upload_limit_and_empty_pdf(client):
    too_large = client.post("/api/candidates/upload", files={"file": ("resume.txt", b"x" * (8 * 1024 * 1024 + 1), "text/plain")})
    assert too_large.status_code == 413, too_large.text
    from pypdf import PdfWriter
    pdf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(pdf)
    scanned = client.post("/api/candidates/upload", files={"file": ("scan.pdf", pdf.getvalue(), "application/pdf")})
    assert scanned.status_code in (400, 422), scanned.text
    assert isinstance(scanned.json()["detail"], str)


def test_demo_is_idempotent_and_matching_uses_real_vacancies(client):
    one = client.post("/api/demo")
    assert one.status_code in (200, 201), one.text
    first = one.json()
    assert len(first["candidates"]) == 4
    assert all(x["is_demo"] for x in first["candidates"])
    again = client.post("/api/demo").json()
    assert again["vacancy_id"] == first["vacancy_id"]
    assert {x["id"] for x in again["candidates"]} == {x["id"] for x in first["candidates"]}
    assert len(client.get("/api/candidates").json()["items"]) == 4
    match = client.post("/api/matches", json={"candidate_id": first["candidates"][0]["id"], "limit": 3})
    assert match.status_code == 200, match.text
    result = match.json()
    assert len(result["items"]) == 3
    assert result["pipeline"]["ml2_available"]
    assert all(x["vacancy"]["id"] for x in result["items"])
    assert all(x["explanation"] for x in result["items"])


def test_feedback_csv_and_formula_injection(client):
    v = vacancy(client)
    c = candidate(client, name="=1+1")
    client.put("/api/reviews", json={"vacancy_id": v["id"], "candidate_id": c["id"], "status": "hold"})
    feedback = client.post("/api/feedback", json={"rating": 4, "comment": "Понятные объяснения", "candidate_id": c["id"]})
    assert feedback.status_code in (200, 201), feedback.text
    assert feedback.json()["id"]
    assert client.get("/api/stats").json()["feedback_count"] == 1
    exported = client.get("/api/export", params={"vacancy_id": v["id"]})
    assert exported.status_code == 200
    assert "text/csv" in exported.headers["content-type"]
    assert "attachment" in exported.headers["content-disposition"]
    content = exported.content.decode("utf-8-sig")
    assert "=1+1" in content
    rows = list(csv.reader(io.StringIO(content), delimiter=";"))
    if len(rows[0]) == 1:
        rows = list(csv.reader(io.StringIO(content)))
    cells = [cell for row in rows for cell in row]
    assert "=1+1" not in cells  # A leading apostrophe neutralizes spreadsheet execution.
    assert "'=1+1" in cells


def test_invalid_inputs_and_unknown_routes_are_json(client):
    assert client.post("/api/candidates", json={"name": "", "text": ""}).status_code == 422
    assert client.post("/api/vacancies", json={"title": ""}).status_code == 422
    assert client.post("/api/feedback", json={"rating": 6}).status_code == 422
    assert client.put("/api/reviews", json={"vacancy_id": "missing", "candidate_id": "missing", "status": "rejected_automatically"}).status_code == 422
    for path in ("/api/does-not-exist", "/api/vacancies/not-a-real-id"):
        r = client.get(path)
        assert r.status_code == 404
        assert "application/json" in r.headers["content-type"]
        assert "detail" in r.json()


def test_weak_skill_evidence_stays_uncertain_until_hr_confirms(client):
    v = vacancy(client, requirements="Python, Kubernetes")
    c = candidate(client, text="Python backend developer. Work experience: 3 years. Kubernetes изучаю, практического опыта пока нет.")
    first = shortlist(client, v["id"])["candidates"][0]
    assert "python" in first["matched_skills"]
    assert "kubernetes" not in first["matched_skills"]
    assert "kubernetes" in first["uncertain_skills"]
    assert first["signals"]["skill_coverage"] < 1
    # Saving a skill as HR-confirmed is a visible human correction, not an
    # automatic promotion of a candidate's uncertain claim.
    changed = client.patch(f"/api/candidates/{c['id']}", json={"skills": c["skills"]})
    assert changed.status_code == 200
    after = shortlist(client, v["id"])["candidates"][0]
    assert "kubernetes" in after["matched_skills"]
    kube = next(row for row in after["evidence"] if row["skill"] == "kubernetes")
    assert kube["source"] == "hr"


def test_unrelated_engineering_vacancy_is_outside_it_scope(client):
    result = client.post("/api/vacancies", json={
        "title": "Civil Engineer",
        "company": "Construction Test",
        "city": "Алматы",
        "requirements": "Construction project supervision, building codes and structural inspection.",
        "description": "Manage construction site safety and concrete quality.",
    })
    assert result.status_code == 422, result.text
    assert isinstance(result.json()["detail"], str)


def test_uncertainty_does_not_mark_machine_learning_or_neighbor_skills(client):
    v = vacancy(client, title="Machine Learning Engineer", requirements="Python, Machine Learning, Docker")
    candidate(client, text="Machine Learning Engineer. Work experience: 3 years. Python developer currently learning Docker.")
    row = shortlist(client, v["id"])["candidates"][0]
    assert "machine learning" in row["matched_skills"]
    assert "python" in row["matched_skills"]
    assert "docker" in row["uncertain_skills"]
    assert "machine learning" not in row["uncertain_skills"]


def test_negative_list_predicate_applies_to_both_skills(client):
    v = vacancy(client, requirements="Python, PostgreSQL, Kubernetes")
    candidate(client, text="Python backend developer. Work experience: 3 years. В PostgreSQL и Kubernetes не работал.")
    row = shortlist(client, v["id"])["candidates"][0]
    assert set(row["uncertain_skills"]) == {"postgresql", "kubernetes"}
    assert "python" in row["matched_skills"]


def test_negative_previous_clause_does_not_leak_across_comma(client):
    v = vacancy(client, requirements="Python, Kubernetes")
    candidate(client, text="Backend developer. Work experience: 3 years. Kubernetes не знаю, Python использую три года.")
    row = shortlist(client, v["id"])["candidates"][0]
    assert "python" in row["matched_skills"]
    assert "kubernetes" in row["uncertain_skills"]


def test_demographics_are_not_skill_evidence_or_professional_source(client):
    no_source = client.post("/api/candidates", json={"name": "Person", "text": "Nationality: Kazakhstan. Date of birth: 2000-01-01"})
    assert no_source.status_code == 422, no_source.text
    v = vacancy(client, title="Java Backend Developer", requirements="Python, Java")
    c = candidate(client, text="Nationality: Python\nMiddle Java developer. Work experience: 3 years.")
    assert "python" not in c["skills"]
    row = shortlist(client, v["id"])["candidates"][0]
    assert "python" not in row["matched_skills"]
    assert "python" in row["missing_skills"]
    assert "java" in row["matched_skills"]
    assert row["signals"]["skill_coverage"] == row["skill_score"]


def test_ml1_failure_reports_actual_query_fallback(client, app):
    class BrokenIndex:
        def match(self, *args, **kwargs):
            raise ValueError("unavailable query model")
    app.state.matching.ml1 = BrokenIndex()
    c = candidate(client)
    result = client.post("/api/matches", json={"candidate_id": c["id"], "limit": 2})
    assert result.status_code == 200, result.text
    pipeline = result.json()["pipeline"]
    assert pipeline["ml1_available"] is False
    assert pipeline["engine"] == "lexical-skills+ml2"
    assert pipeline["warning"]
    assert len(result.json()["items"]) == 2


def test_pdf_stream_decompression_is_bounded(client):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    stream = DecodedStreamObject()
    stream.set_data(b"q " * 6_000_001)
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    output = io.BytesIO()
    writer.write(output)
    assert len(output.getvalue()) < 100_000
    response = client.post("/api/candidates/upload", files={"file": ("compressed.pdf", output.getvalue(), "application/pdf")})
    assert response.status_code == 422, response.text
    assert client.get("/api/candidates").json()["items"] == []
