"""SQLite persistence. Every private row is selected together with its session ID."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3


class Store:
    def __init__(self, path):
        self.path = Path(path)
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
            """)

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
        return json.loads(row[0]) if row else None

    def all(self, table, sid):
        assert table in {"candidates", "vacancies", "feedback"}
        with self.connect() as db:
            rows = db.execute(f"SELECT data FROM {table} WHERE session_id=? ORDER BY rowid DESC", (sid,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put(self, table, sid, record):
        assert table in {"candidates", "vacancies", "feedback"}
        with self.connect() as db:
            db.execute(f"INSERT INTO {table}(id, session_id, data) VALUES (?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET data=excluded.data WHERE session_id=excluded.session_id",
                       (record["id"], sid, json.dumps(record, ensure_ascii=False)))

    def delete_candidate(self, sid, identity):
        with self.connect() as db:
            db.execute("DELETE FROM candidates WHERE session_id=? AND id=?", (sid, identity))
            db.execute("DELETE FROM reviews WHERE session_id=? AND candidate_id=?", (sid, identity))
            # Candidate-linked feedback is removed alongside the candidate.
            for row in db.execute("SELECT id, data FROM feedback WHERE session_id=?", (sid,)).fetchall():
                if json.loads(row["data"]).get("candidate_id") == identity:
                    db.execute("DELETE FROM feedback WHERE id=? AND session_id=?", (row["id"], sid))

    def review(self, sid, vacancy_id, candidate_id, status):
        with self.connect() as db:
            db.execute("INSERT INTO reviews VALUES (?,?,?,?) ON CONFLICT(session_id,vacancy_id,candidate_id) "
                       "DO UPDATE SET status=excluded.status", (sid, vacancy_id, candidate_id, status))

    def reviews(self, sid, vacancy_id=None):
        with self.connect() as db:
            rows = db.execute("SELECT vacancy_id,candidate_id,status FROM reviews WHERE session_id=?" +
                              (" AND vacancy_id=?" if vacancy_id else ""),
                              (sid, vacancy_id) if vacancy_id else (sid,)).fetchall()
        return [dict(row) for row in rows]

    def clear_session(self, sid):
        with self.connect() as db:
            for table in ("candidates", "vacancies", "reviews", "feedback"):
                db.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
