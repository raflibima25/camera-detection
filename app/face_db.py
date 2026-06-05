import sqlite3
from pathlib import Path

import numpy as np

from app import config

DB_PATH = Path("data/faces.db")


class FaceDB:
    def __init__(self) -> None:
        DB_PATH.parent.mkdir(exist_ok=True)
        self._con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._init_db()
        self._embeddings: list[tuple[int, str, np.ndarray]] = []
        self._mtime: float = 0.0
        self._reload()

    def _init_db(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._con.commit()

    def _reload(self) -> None:
        try:
            mtime = DB_PATH.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        # Reopen connection to get fresh data from disk
        self._con.close()
        self._con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._init_db()
        rows = self._con.execute(
            "SELECT id, name, embedding FROM faces"
        ).fetchall()
        self._embeddings = [
            (row[0], row[1], np.frombuffer(row[2], dtype=np.float32))
            for row in rows
        ]
        print(f"[INFO] Face DB reloaded: {len(self._embeddings)} faces")

    def check_reload(self) -> None:
        """Call from main loop every few seconds."""
        self._reload()

    # --- Write ---

    def enroll(self, name: str, embedding: np.ndarray) -> int:
        blob = embedding.astype(np.float32).tobytes()
        cur = self._con.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, blob)
        )
        self._con.commit()
        self._reload()
        return cur.lastrowid

    def delete(self, face_id: int) -> bool:
        cur = self._con.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self._con.commit()
        self._reload()
        return cur.rowcount > 0

    def delete_by_name(self, name: str) -> int:
        """Delete all embeddings for one name. Return number of rows deleted."""
        cur = self._con.execute("DELETE FROM faces WHERE name = ?", (name,))
        self._con.commit()
        self._reload()
        return cur.rowcount

    # --- Read ---

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        Return (name, similarity). If nothing exceeds the threshold
        or the DB is empty, return ("Unknown", best_score).
        """
        if not self._embeddings:
            return "Unknown", 0.0

        query = embedding.astype(np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-9)

        best_name, best_score = "Unknown", 0.0
        for _, name, stored in self._embeddings:
            stored_norm = stored / (np.linalg.norm(stored) + 1e-9)
            score = float(np.dot(query_norm, stored_norm))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < config.FACE_THRESHOLD:
            best_name = "Unknown"

        return best_name, best_score

    def list_all(self) -> list[tuple[int, str, str]]:
        rows = self._con.execute(
            "SELECT id, name, created_at FROM faces ORDER BY id"
        ).fetchall()
        return rows

    def list_grouped(self) -> list[tuple[str, int, str]]:
        """Return [(name, embedding_count, earliest_created_at)] sorted by earliest enrollment."""
        rows = self._con.execute(
            """
            SELECT name, COUNT(*) AS cnt, MIN(created_at) AS first_enrolled
            FROM faces
            GROUP BY name
            ORDER BY first_enrolled
            """
        ).fetchall()
        return rows
