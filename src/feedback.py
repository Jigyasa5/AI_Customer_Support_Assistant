# src/feedback.py

import sqlite3
from datetime import datetime


DATABASE = "feedback.db"


def create_feedback_table():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_feedback(
    question,
    answer,
    feedback
):
    """
    Store user feedback.

    feedback should be:
        👍
        👎
    """

    if feedback not in ["👍", "👎"]:

        raise ValueError(
            "Feedback must be 👍 or 👎"
        )

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO feedback
        (
            question,
            answer,
            feedback,
            timestamp
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            question,
            answer,
            feedback,
            timestamp
        )
    )

    connection.commit()
    connection.close()


def get_feedback():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            question,
            answer,
            feedback,
            timestamp
        FROM feedback
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def feedback_statistics():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT feedback, COUNT(*)
        FROM feedback
        GROUP BY feedback
        """
    )

    statistics = cursor.fetchall()

    connection.close()

    return statistics


if __name__ == "__main__":

    create_feedback_table()

    save_feedback(
        "How can I get a refund?",
        "You can request a refund within 7 days.",
        "👍"
    )

    save_feedback(
        "Where is my order?",
        "I don't have enough information.",
        "👎"
    )

    print("\nFeedback:")
    
    for row in get_feedback():
        print(row)

    print("\nStatistics:")

    for row in feedback_statistics():
        print(row)