import sqlite3

DATABASE = "users.db"

def create_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (username, password)
            VALUES (?, ?)
            """,
            (username, password)
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, password
        FROM users
        WHERE username = ?
        """,
        (username,)
    )
    user = cursor.fetchone()
    conn.close()
    return user

def save_chat(username, question, answer):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history
        (username, question, answer)
        VALUES (?, ?, ?)
        """,
        (username, question, answer)
    )

    conn.commit()
    conn.close()


def get_chat_history(username):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT question, answer, created_at
        FROM chat_history
        WHERE username = ?
        ORDER BY id ASC
        """,
        (username,)
    )

    history = cursor.fetchall()
    conn.close()
    return history