import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")

def init_db(db_path: str = DB_PATH):
    """Initializes the SQLite database with user-specific chat history table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        sources_json TEXT,
        verification_json TEXT,
        timestamp TEXT NOT NULL
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_time ON chat_history (username, timestamp)")
    conn.commit()
    conn.close()

def _clean_for_json(obj):
    """Recursively converts numpy types (bool_, float64, int64) into native Python types for JSON serialization."""
    if hasattr(obj, "item"):  # numpy scalar types (np.bool_, np.float64, np.int64, etc.)
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)

def save_chat_turn(username: str, question: str, answer: str, sources: list, verification: list, db_path: str = DB_PATH) -> int:
    """Saves a single Q&A turn with serialized sources and verification records."""
    if not username:
        username = "default_user"
        
    # Serialize sources
    serialized_sources = []
    if sources:
        for doc in sources:
            if hasattr(doc, "metadata"):
                serialized_sources.append({
                    "source_file": str(doc.metadata.get("source_file", "Document")),
                    "page": str(doc.metadata.get("page", "?")),
                    "page_content": str(getattr(doc, "page_content", ""))[:350]
                })
            elif isinstance(doc, dict):
                serialized_sources.append({k: _clean_for_json(v) for k, v in doc.items()})
                
    # Clean verification records (convert numpy.bool_ and numpy.float64 to Python native types)
    serialized_verification = []
    if verification:
        for item in verification:
            if isinstance(item, dict):
                serialized_verification.append({
                    "sentence": str(item.get("sentence", "")),
                    "supported": bool(item.get("supported", False)),
                    "similarity": float(item.get("similarity", 0.0)),
                    "quote": str(item.get("quote", "")),
                    "cited_page": str(item.get("cited_page", ""))
                })
            else:
                serialized_verification.append(_clean_for_json(item))
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO chat_history (username, question, answer, sources_json, verification_json, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        username,
        str(question),
        str(answer),
        json.dumps(serialized_sources, ensure_ascii=False),
        json.dumps(serialized_verification, ensure_ascii=False),
        timestamp
    ))
    last_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return last_id

def load_user_history(username: str, db_path: str = DB_PATH) -> list:
    """Loads all chat turns for a specific user, ordered chronologically."""
    if not username:
        return []
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, question, answer, sources_json, verification_json, timestamp
    FROM chat_history
    WHERE username = ?
    ORDER BY id ASC
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        turn_id, question, answer, s_json, v_json, ts = r
        try:
            sources_data = json.loads(s_json) if s_json else []
        except Exception:
            sources_data = []
        try:
            verification_data = json.loads(v_json) if v_json else []
        except Exception:
            verification_data = []
            
        history.append({
            "id": turn_id,
            "question": question,
            "answer": answer,
            "sources": sources_data,
            "verification": verification_data,
            "timestamp": ts
        })
    return history

def clear_user_history(username: str, db_path: str = DB_PATH):
    """Deletes all persistent chat history for a specific user."""
    if not username:
        return
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE username = ?", (username,))
    conn.commit()
    conn.close()
