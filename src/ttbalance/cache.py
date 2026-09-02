"""Persistent store of every evaluation ever run.

API calls are the scarce resource in this competition, so nothing is thrown
away: repeated observations of the same configuration accumulate and are
averaged, which is how we fight the noise in low-fidelity runs.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

from .spec import Params, canonical

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    game     TEXT NOT NULL,
    key      TEXT NOT NULL,
    run_type TEXT NOT NULL,
    score    REAL NOT NULL,
    ts       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS obs_lookup ON observations (game, run_type, key);
CREATE TABLE IF NOT EXISTS params (
    key    TEXT PRIMARY KEY,
    game   TEXT NOT NULL,
    params TEXT NOT NULL
);
"""


class Cache:
    def __init__(self, path: str = "results/cache.sqlite"):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        # A background search and an interactive command routinely hold this
        # file open at the same time; WAL plus a busy timeout keeps a
        # concurrent reader from turning into "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, game: str, params: Params, run_type: str, score: float) -> None:
        key = canonical(params)
        with self._lock:
            self._conn.execute(
                "INSERT INTO observations (game, key, run_type, score, ts)"
                " VALUES (?,?,?,?,?)", (game, key, run_type, score, time.time()))
            self._conn.execute(
                "INSERT OR IGNORE INTO params (key, game, params) VALUES (?,?,?)",
                (key, game, key))
            self._conn.commit()

    def scores(self, game: str, params: Params, run_type: str) -> List[float]:
        key = canonical(params)
        with self._lock:
            rows = self._conn.execute(
                "SELECT score FROM observations WHERE game=? AND run_type=? AND key=?",
                (game, run_type, key)).fetchall()
        return [r[0] for r in rows]

    def best(self, game: str, run_type: Optional[str] = None,
             limit: int = 20, min_obs: int = 1) -> List[Tuple[str, float, int, str]]:
        """Top configurations by mean score: (key, mean, n_obs, run_type)."""
        sql = ("SELECT key, AVG(score), COUNT(*), run_type FROM observations"
               " WHERE game=?")
        args: List[object] = [game]
        if run_type:
            sql += " AND run_type=?"
            args.append(run_type)
        sql += (" GROUP BY key, run_type HAVING COUNT(*)>=? ORDER BY AVG(score) DESC"
                " LIMIT ?")
        args += [min_obs, limit]
        with self._lock:
            return list(self._conn.execute(sql, args).fetchall())

    def stats(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT game, run_type, COUNT(*) FROM observations"
                " GROUP BY game, run_type").fetchall()
        out: Dict[str, Dict[str, int]] = {}
        for game, run_type, n in rows:
            out.setdefault(game, {})[run_type] = n
        return out
