"""
Session Manager Module
Handles creation, naming, storage, and retrieval of analysis sessions.
Each session contains multiple image analyses with aggregate statistics.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "visionmeasure.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_session_tables():
    """Create session-related tables."""
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            total_images INTEGER DEFAULT 0,
            successful INTEGER DEFAULT 0,
            failed INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS session_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            filename TEXT,
            relative_path TEXT DEFAULT '',
            image_width INTEGER,
            image_height INTEGER,
            filesize INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1,
            error_message TEXT DEFAULT '',
            strategy_used TEXT DEFAULT '',
            objects_detected INTEGER DEFAULT 0,
            measurements_json TEXT DEFAULT '[]',
            shape_classes_json TEXT DEFAULT '[]',
            quality_json TEXT DEFAULT '{}',
            processing_time REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.commit()
    conn.close()


def create_session(user_id, name, description=""):
    """Create a new named session. Returns session_id."""
    conn = _conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (user_id, name, description) VALUES (?, ?, ?)",
        (user_id, name.strip(), description.strip()),
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def save_session_result(session_id, result_data):
    """
    Save a single image result to a session.

    result_data keys:
        filename, relative_path, image_width, image_height, filesize,
        success, error_message, strategy_used, objects_detected,
        measurements, shape_classes, quality, processing_time
    """
    conn = _conn()
    cursor = conn.cursor()

    measurements_json = json.dumps(result_data.get("measurements", []), default=str)
    shapes_json = json.dumps(result_data.get("shape_classes", []), default=str)
    quality_json = json.dumps(result_data.get("quality", {}), default=str)

    cursor.execute(
        """INSERT INTO session_results
        (session_id, filename, relative_path, image_width, image_height, filesize,
         success, error_message, strategy_used, objects_detected,
         measurements_json, shape_classes_json, quality_json, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            result_data.get("filename", ""),
            result_data.get("relative_path", ""),
            result_data.get("image_width", 0),
            result_data.get("image_height", 0),
            result_data.get("filesize", 0),
            1 if result_data.get("success", True) else 0,
            result_data.get("error_message", ""),
            result_data.get("strategy_used", ""),
            result_data.get("objects_detected", 0),
            measurements_json,
            shapes_json,
            quality_json,
            result_data.get("processing_time", 0),
        ),
    )
    conn.commit()
    conn.close()


def update_session_stats(session_id, total, successful, failed, skipped):
    """Update session aggregate counts."""
    conn = _conn()
    conn.execute(
        """UPDATE sessions
        SET total_images=?, successful=?, failed=?, skipped=?,
            status='completed', updated_at=datetime('now')
        WHERE id=?""",
        (total, successful, failed, skipped, session_id),
    )
    conn.commit()
    conn.close()


def get_user_sessions(user_id, limit=50):
    """Get all sessions for a user, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_detail(session_id):
    """Get session info + all results."""
    conn = _conn()
    session = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if session is None:
        conn.close()
        return None, []

    results = conn.execute(
        "SELECT * FROM session_results WHERE session_id=? ORDER BY filename",
        (session_id,),
    ).fetchall()
    conn.close()

    session_dict = dict(session)
    results_list = []
    for r in results:
        rd = dict(r)
        rd["measurements"] = json.loads(rd.get("measurements_json", "[]"))
        rd["shape_classes"] = json.loads(rd.get("shape_classes_json", "[]"))
        rd["quality"] = json.loads(rd.get("quality_json", "{}"))
        results_list.append(rd)

    return session_dict, results_list


def get_session_statistics(session_id):
    """
    Compute aggregate statistics for a session.
    Returns dict with counts, area stats, size distributions, quality metrics, etc.
    """
    _, results = get_session_detail(session_id)

    if not results:
        return None

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    # Aggregate measurements
    all_objects = []
    all_areas = []
    all_widths = []
    all_heights = []
    all_perimeters = []
    strategies_used = {}
    shapes_found = {}
    quality_scores = []
    processing_times = []

    for r in successful:
        processing_times.append(r.get("processing_time", 0))

        strategy = r.get("strategy_used", "Unknown")
        strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

        for m in r.get("measurements", []):
            all_objects.append(m)
            if isinstance(m, dict):
                if m.get("area_cm2") is not None:
                    all_areas.append(m["area_cm2"])
                    all_widths.append(m.get("width_cm", 0))
                    all_heights.append(m.get("height_cm", 0))
                    all_perimeters.append(m.get("perimeter_cm", 0))
                elif m.get("area_px") is not None:
                    all_areas.append(m["area_px"])
                    all_widths.append(m.get("width_px", 0))
                    all_heights.append(m.get("height_px", 0))
                    all_perimeters.append(m.get("perimeter_px", 0))

        for s in r.get("shape_classes", []):
            if isinstance(s, dict):
                shape = s.get("shape", "Unknown")
            else:
                shape = str(s)
            shapes_found[shape] = shapes_found.get(shape, 0) + 1

        q = r.get("quality", {})
        if isinstance(q, dict) and "score" in q:
            quality_scores.append(q["score"])

    import numpy as np

    stats = {
        "total_images": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_objects_detected": len(all_objects),
        "avg_objects_per_image": len(all_objects) / max(len(successful), 1),
        "strategies_used": strategies_used,
        "shapes_found": shapes_found,
        "total_processing_time": sum(processing_times),
        "avg_processing_time": np.mean(processing_times) if processing_times else 0,
    }

    if all_areas:
        stats["area_stats"] = {
            "min": float(np.min(all_areas)),
            "max": float(np.max(all_areas)),
            "mean": float(np.mean(all_areas)),
            "median": float(np.median(all_areas)),
            "std": float(np.std(all_areas)),
        }
        stats["width_stats"] = {
            "min": float(np.min(all_widths)),
            "max": float(np.max(all_widths)),
            "mean": float(np.mean(all_widths)),
            "std": float(np.std(all_widths)),
        }
        stats["all_areas"] = [float(a) for a in all_areas]
        stats["all_widths"] = [float(w) for w in all_widths]
        stats["all_heights"] = [float(h) for h in all_heights]

    if quality_scores:
        stats["avg_quality_score"] = float(np.mean(quality_scores))

    return stats


def delete_session(session_id):
    """Delete a session and all its results."""
    conn = _conn()
    conn.execute("DELETE FROM session_results WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()


# Initialize on import
init_session_tables()
