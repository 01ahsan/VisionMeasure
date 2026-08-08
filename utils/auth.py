"""
Authentication & Database Module
- bcrypt with per-user salts
- Login attempt rate limiting
- Session token expiry
- WAL mode for better SQLite concurrency (Fix #8)
"""

import sqlite3
import bcrypt
import os
import json
import time
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "visionmeasure.db")

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
SESSION_EXPIRY_HOURS = 24


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for concurrent reads (Fix #8)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            institution TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username_or_email TEXT,
            ip_hint TEXT DEFAULT '',
            success INTEGER DEFAULT 0,
            attempted_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            image_width INTEGER,
            image_height INTEGER,
            objects_detected INTEGER,
            measurements_json TEXT,
            settings_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS guest_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            used_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def _hash_password(password):
    """Hash password with bcrypt (auto-generates per-user salt)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password, stored_hash):
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


def _check_rate_limit(username_or_email):
    """
    Check if login attempts are rate-limited.
    Returns (is_locked, remaining_seconds).
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cutoff = (datetime.now() - timedelta(minutes=LOCKOUT_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM login_attempts WHERE username_or_email=? AND success=0 AND attempted_at > ?",
        (username_or_email, cutoff),
    )
    count = cursor.fetchone()["cnt"]
    conn.close()

    if count >= MAX_LOGIN_ATTEMPTS:
        return True, LOCKOUT_MINUTES * 60
    return False, 0


def _record_login_attempt(username_or_email, success):
    conn = _get_connection()
    conn.execute(
        "INSERT INTO login_attempts (username_or_email, success) VALUES (?, ?)",
        (username_or_email, 1 if success else 0),
    )
    if success:
        # Clear failed attempts on successful login
        conn.execute(
            "DELETE FROM login_attempts WHERE username_or_email=? AND success=0",
            (username_or_email,),
        )
    conn.commit()
    conn.close()


def signup(username, email, password, full_name="", institution=""):
    """Register a new user with bcrypt hashing."""
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = ?", (username.strip(),))
    if cursor.fetchone():
        conn.close()
        return False, "Username already taken. Please choose another."

    cursor.execute("SELECT id FROM users WHERE email = ?", (email.strip().lower(),))
    if cursor.fetchone():
        conn.close()
        return False, "This email is already registered. Try signing in instead."

    try:
        hashed = _hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, full_name, institution) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), hashed, full_name.strip(), institution.strip()),
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Registration failed. Please try again."
    finally:
        conn.close()


def login(username_or_email, password):
    """Authenticate with rate limiting and bcrypt verification."""
    # Check rate limit
    is_locked, wait_time = _check_rate_limit(username_or_email)
    if is_locked:
        mins = LOCKOUT_MINUTES
        return False, f"Too many failed attempts. Account locked for {mins} minutes."

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE (username = ? OR email = ?) AND is_active = 1",
        (username_or_email, username_or_email.lower()),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        user_dict = dict(row)
        if _verify_password(password, user_dict["password_hash"]):
            _record_login_attempt(username_or_email, True)
            return True, user_dict
        else:
            _record_login_attempt(username_or_email, False)
            return False, "Invalid password."
    else:
        _record_login_attempt(username_or_email, False)
        return False, "Account not found."


def get_user_analysis_count(user_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM analysis_history WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()["count"]
    conn.close()
    return result


def save_analysis(user_id, filename, width, height, objects_detected, measurements, settings):
    conn = _get_connection()
    conn.execute(
        "INSERT INTO analysis_history (user_id, filename, image_width, image_height, objects_detected, measurements_json, settings_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, filename, width, height, objects_detected, json.dumps(measurements, default=str), json.dumps(settings, default=str)),
    )
    conn.commit()
    conn.close()


def get_analysis_history(user_id, limit=20):
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_guest_usage_count(session_id):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM guest_usage WHERE session_id = ?", (session_id,))
    result = cursor.fetchone()["count"]
    conn.close()
    return result


def record_guest_usage(session_id):
    conn = _get_connection()
    conn.execute("INSERT INTO guest_usage (session_id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()


init_db()
