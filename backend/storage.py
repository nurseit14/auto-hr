"""SQLite persistence. Every private row is selected together with its session ID."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3


class Store:
    def __init__(self, path, catalogue_ids=()):
        self.path = Path(path)
        self.catalogue_ids = frozenset(catalogue_ids)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS candidate_sessions ON candidates(session_id);
                CREATE TABLE IF NOT EXISTS vacancies (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS vacancy_sessions ON vacancies(session_id);
                CREATE TABLE IF NOT EXISTS reviews (
                    session_id TEXT NOT NULL, vacancy_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL, status TEXT NOT NULL,
                    PRIMARY KEY(session_id, vacancy_id, candidate_id));
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS candidate_vacancies (
                    session_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    vacancy_id TEXT NOT NULL,
                    PRIMARY KEY(session_id, candidate_id, vacancy_id));
                CREATE INDEX IF NOT EXISTS vacancy_candidates
                    ON candidate_vacancies(session_id, vacancy_id);
                CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY);
            """)
            # A previous HR decision is explicit evidence of an application.
            # Merely having scored a CV against a vacancy is deliberately not.
            db.execute("BEGIN IMMEDIATE")
            migration = "candidate-vacancies-v1"
            if not db.execute("SELECT 1 FROM schema_migrations WHERE name=?", (migration,)).fetchone():
                rows = db.execute("""
                    SELECT r.session_id, r.candidate_id, r.vacancy_id,
                           v.id AS custom_vacancy_id
                    FROM reviews r
                    JOIN candidates c ON c.id=r.candidate_id AND c.session_id=r.session_id
                    LEFT JOIN vacancies v ON v.id=r.vacancy_id AND v.session_id=r.session_id
                """).fetchall()
                for row in rows:
                    if row["custom_vacancy_id"] or row["vacancy_id"] in self.catalogue_ids:
                        db.execute("INSERT OR IGNORE INTO candidate_vacancies VALUES (?,?,?)",
                                   (row["session_id"], row["candidate_id"], row["vacancy_id"]))
                db.execute("INSERT INTO schema_migrations(name) VALUES (?)", (migration,))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def session_exists(self, sid):
        with self.connect() as db:
            return bool(db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone())

    def create_session(self, sid):
        with self.connect() as db:
            db.execute("INSERT INTO sessions(id) VALUES (?)", (sid,))

    def get(self, table, sid, identity):
        assert table in {"candidates", "vacancies"}
        with self.connect() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE session_id=? AND id=?", (sid, identity)).fetchone()
            record = json.loads(row[0]) if row else None
            if record is not None and table == "candidates":
                record["vacancy_ids"] = [link[0] for link in db.execute(
                    "SELECT vacancy_id FROM candidate_vacancies WHERE session_id=? AND candidate_id=? ORDER BY vacancy_id",
                    (sid, identity)).fetchall()]
        return record

    def all(self, table, sid):
        assert table in {"candidates", "vacancies", "feedback"}
        with self.connect() as db:
            rows = db.execute(f"SELECT data FROM {table} WHERE session_id=? ORDER BY rowid DESC", (sid,)).fetchall()
            records = [json.loads(row[0]) for row in rows]
            if table == "candidates":
                links = {}
                for link in db.execute("SELECT candidate_id,vacancy_id FROM candidate_vacancies WHERE session_id=? ORDER BY vacancy_id", (sid,)):
                    links.setdefault(link["candidate_id"], []).append(link["vacancy_id"])
                for record in records:
                    record["vacancy_ids"] = links.get(record["id"], [])
        return records

    def put(self, table, sid, record, vacancy_ids=None):
        assert table in {"candidates", "vacancies", "feedback"}
        with self.connect() as db:
            # Links have their own source of truth, never an embedded JSON copy.
            stored = {key: value for key, value in record.items() if table != "candidates" or key != "vacancy_ids"}
            db.execute(f"INSERT INTO {table}(id, session_id, data) VALUES (?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET data=excluded.data WHERE session_id=excluded.session_id",
                       (record["id"], sid, json.dumps(stored, ensure_ascii=False)))
            if vacancy_ids is not None:
                assert table == "candidates"
                self._set_candidate_vacancies(db, sid, record["id"], vacancy_ids)

    def _set_candidate_vacancies(self, db, sid, candidate_id, vacancy_ids):
        if not db.execute("SELECT 1 FROM candidates WHERE session_id=? AND id=?", (sid, candidate_id)).fetchone():
            raise LookupError("Кандидат не найден в этом рабочем пространстве.")
        vacancy_ids = sorted(set(vacancy_ids))
        for vacancy_id in vacancy_ids:
            if vacancy_id not in self.catalogue_ids and not db.execute(
                    "SELECT 1 FROM vacancies WHERE session_id=? AND id=?", (sid, vacancy_id)).fetchone():
                raise LookupError("Вакансия не найдена в этом рабочем пространстве.")
        if vacancy_ids:
            parameters = ",".join("?" for _ in vacancy_ids)
            for table in ("candidate_vacancies", "reviews"):
                db.execute(f"DELETE FROM {table} WHERE session_id=? AND candidate_id=? AND vacancy_id NOT IN ({parameters})",
                           (sid, candidate_id, *vacancy_ids))
            db.executemany("INSERT OR IGNORE INTO candidate_vacancies VALUES (?,?,?)",
                           ((sid, candidate_id, vacancy_id) for vacancy_id in vacancy_ids))
        else:
            for table in ("candidate_vacancies", "reviews"):
                db.execute(f"DELETE FROM {table} WHERE session_id=? AND candidate_id=?", (sid, candidate_id))

    def set_candidate_vacancies(self, sid, candidate_id, vacancy_ids):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._set_candidate_vacancies(db, sid, candidate_id, vacancy_ids)

    def candidates_for_vacancy(self, sid, vacancy_id):
        return [row for row in self.all("candidates", sid) if vacancy_id in row["vacancy_ids"]]

    def delete_vacancy(self, sid, identity):
        with self.connect() as db:
            db.execute("DELETE FROM vacancies WHERE session_id=? AND id=?", (sid, identity))
            db.execute("DELETE FROM candidate_vacancies WHERE session_id=? AND vacancy_id=?", (sid, identity))
            db.execute("DELETE FROM reviews WHERE session_id=? AND vacancy_id=?", (sid, identity))
            for row in db.execute("SELECT id, data FROM feedback WHERE session_id=?", (sid,)).fetchall():
                if json.loads(row["data"]).get("vacancy_id") == identity:
                    db.execute("DELETE FROM feedback WHERE id=? AND session_id=?", (row["id"], sid))

    def delete_candidate(self, sid, identity):
        with self.connect() as db:
            db.execute("DELETE FROM candidates WHERE session_id=? AND id=?", (sid, identity))
            db.execute("DELETE FROM candidate_vacancies WHERE session_id=? AND candidate_id=?", (sid, identity))
            db.execute("DELETE FROM reviews WHERE session_id=? AND candidate_id=?", (sid, identity))
            # Candidate-linked feedback is removed alongside the candidate.
            for row in db.execute("SELECT id, data FROM feedback WHERE session_id=?", (sid,)).fetchall():
                if json.loads(row["data"]).get("candidate_id") == identity:
                    db.execute("DELETE FROM feedback WHERE id=? AND session_id=?", (row["id"], sid))

    def review(self, sid, vacancy_id, candidate_id, status):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM candidate_vacancies WHERE session_id=? AND vacancy_id=? AND candidate_id=?",
                              (sid, vacancy_id, candidate_id)).fetchone():
                raise LookupError("Кандидат больше не добавлен в эту вакансию. Обновите список и добавьте его из базы резюме.")
            db.execute("INSERT INTO reviews VALUES (?,?,?,?) ON CONFLICT(session_id,vacancy_id,candidate_id) "
                       "DO UPDATE SET status=excluded.status", (sid, vacancy_id, candidate_id, status))

    def reviews(self, sid, vacancy_id=None):
        with self.connect() as db:
            rows = db.execute("""SELECT r.vacancy_id,r.candidate_id,r.status FROM reviews r
                              JOIN candidate_vacancies a ON a.session_id=r.session_id
                              AND a.vacancy_id=r.vacancy_id AND a.candidate_id=r.candidate_id
                              JOIN candidates c ON c.session_id=r.session_id AND c.id=r.candidate_id
                              WHERE r.session_id=?""" +
                              (" AND r.vacancy_id=?" if vacancy_id else ""),
                              (sid, vacancy_id) if vacancy_id else (sid,)).fetchall()
        return [dict(row) for row in rows]

    def clear_session(self, sid):
        with self.connect() as db:
            for table in ("candidate_vacancies", "candidates", "vacancies", "reviews", "feedback"):
                db.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
