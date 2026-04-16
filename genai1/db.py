import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_NAME = os.getenv("POSTGRES_DB", "postgres")   # FIXED
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password123")

conn = None

# ==================== DB INIT ====================
def init_db():
    """Initialize PostgreSQL connection and create chat table with feedback"""
    global conn
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            refs JSONB,
            confidence REAL DEFAULT 0.0,
            rating INT CHECK (rating BETWEEN 1 AND 5),
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

# ==================== CHAT FUNCTIONS ====================
def save_chat(question: str, answer: str, references: list,
              confidence: float = 0.0, rating: int = None, comment: str = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_history (question, answer, refs, confidence, rating, comment)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (question, answer, json.dumps(references), confidence, rating, comment)
        )

def load_chat():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT question, answer, refs, confidence, rating, comment, created_at
            FROM chat_history
            ORDER BY id
        """)
        rows = cur.fetchall()

    for row in rows:
        row["refs"] = row["refs"] or []
    return rows

def clear_chat():
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chat_history")

# ==================== FEEDBACK FUNCTIONS ====================
def save_feedback(question: str, answer: str, rating: int, comment: str = ""):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE chat_history
            SET rating = %s, comment = %s
            WHERE question = %s AND answer = %s
            """,
            (rating, comment, question, answer)
        )

def load_feedback():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT question, answer, rating, comment, created_at
            FROM chat_history
            WHERE rating IS NOT NULL
            ORDER BY created_at DESC
        """)
        return cur.fetchall()