"""RAG Memory module — SQLite FTS5 based memory for the KZ Admission Bot."""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store", "memory.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # Main memory table
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # FTS5 virtual table
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            category,
            content=memories,
            content_rowid=id,
            tokenize='unicode61'
        )
    """)

    # Triggers for FTS sync
    c.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category)
            VALUES (new.id, new.content, new.category);
        END
    """)
    c.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category)
            VALUES ('delete', old.id, old.content, old.category);
        END
    """)
    c.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category)
            VALUES ('delete', old.id, old.content, old.category);
            INSERT INTO memories_fts(rowid, content, category)
            VALUES (new.id, new.content, new.category);
        END
    """)

    # Documents table
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            doc_type TEXT NOT NULL,
            file_name TEXT,
            extracted_text TEXT,
            status TEXT DEFAULT 'uploaded',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # User profiles
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            city TEXT,
            interests TEXT,
            budget TEXT,
            direction TEXT,
            target_university TEXT,
            onboarded INTEGER DEFAULT 0,
            onboard_step TEXT DEFAULT 'start',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


def store_memory(user_id: int, content: str, category: str = "general", metadata: dict = None):
    conn = get_connection()
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    conn.execute(
        "INSERT INTO memories (user_id, category, content, metadata) VALUES (?, ?, ?, ?)",
        (user_id, category, content, meta_json)
    )
    conn.commit()
    conn.close()


def search_memories(user_id: int, query: str, limit: int = 10) -> List[Dict]:
    conn = get_connection()
    # FTS5 search
    rows = conn.execute("""
        SELECT m.id, m.category, m.content, m.metadata, m.created_at,
               rank
        FROM memories_fts fts
        JOIN memories m ON m.id = fts.rowid
        WHERE memories_fts MATCH ?
        AND m.user_id = ?
        ORDER BY rank
        LIMIT ?
    """, (query, user_id, limit)).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "category": row["category"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "created_at": row["created_at"],
        })
    conn.close()
    return results


def get_recent_memories(user_id: int, limit: int = 20) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, category, content, metadata, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "category": row["category"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "created_at": row["created_at"],
        })
    conn.close()
    return results


def get_user_profile(user_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def upsert_user_profile(user_id: int, **kwargs):
    conn = get_connection()
    existing = conn.execute("SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if existing:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [user_id]
        conn.execute(f"UPDATE user_profiles SET {sets}, updated_at = datetime('now') WHERE user_id = ?", vals)
    else:
        kwargs["user_id"] = user_id
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join("?" for _ in kwargs)
        conn.execute(f"INSERT INTO user_profiles ({cols}) VALUES ({placeholders})", list(kwargs.values()))
    conn.commit()
    conn.close()


def get_document_checklist(user_id: int) -> List[Dict]:
    """Return document checklist with upload status."""
    required_docs = [
        {"type": "passport_rf", "name": "Паспорт РФ (копия)"},
        {"type": "attestat", "name": "Аттестат о среднем образовании"},
        {"type": "attestat_prilozhenie", "name": "Приложение к аттестату (с оценками)"},
        {"type": "attestat_translation", "name": "Нотариальный перевод аттестата на каз/рус"},
        {"type": "attestat_apostille", "name": "Апостиль на аттестат"},
        {"type": "nostrification", "name": "Нострификация аттестата"},
        {"type": "medical_cert", "name": "Медицинская справка (форма 075/у или 086/у)"},
        {"type": "photos", "name": "Фото 3x4 (6 штук)"},
        {"type": "ent_cert", "name": "Сертификат ЕНТ (если требуется)"},
        {"type": "application_form", "name": "Заявление на поступление"},
        {"type": "military_doc", "name": "Приписное свидетельство / военный билет"},
        {"type": "birth_cert", "name": "Свидетельство о рождении (копия)"},
    ]

    conn = get_connection()
    uploaded = conn.execute(
        "SELECT doc_type, file_name, created_at FROM documents WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    conn.close()

    uploaded_types = {row["doc_type"]: {"file_name": row["file_name"], "date": row["created_at"]} for row in uploaded}

    checklist = []
    for doc in required_docs:
        info = uploaded_types.get(doc["type"])
        checklist.append({
            "type": doc["type"],
            "name": doc["name"],
            "uploaded": info is not None,
            "file_name": info["file_name"] if info else None,
            "date": info["date"] if info else None,
        })
    return checklist


def save_document(user_id: int, doc_type: str, file_name: str, extracted_text: str = ""):
    conn = get_connection()
    # Check if already exists — update
    existing = conn.execute(
        "SELECT id FROM documents WHERE user_id = ? AND doc_type = ?",
        (user_id, doc_type)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE documents SET file_name = ?, extracted_text = ?, status = 'uploaded' WHERE id = ?",
            (file_name, extracted_text, existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO documents (user_id, doc_type, file_name, extracted_text) VALUES (?, ?, ?, ?)",
            (user_id, doc_type, file_name, extracted_text)
        )
    conn.commit()
    conn.close()

    # Also store extracted text in memories for RAG
    if extracted_text.strip():
        store_memory(user_id, f"Документ '{file_name}' ({doc_type}): {extracted_text[:2000]}", category="document")


# Initialize DB on import
init_db()
