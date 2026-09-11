"""
Mock internal systems: an orders database and an escalation ticket queue.

In a real deployment these functions would call out to your order management
system (e.g. Shopify, an internal REST API) and your ticketing system (e.g.
Zendesk, Jira Service Desk). Here we simulate both with a local SQLite file so
the whole project runs with zero external dependencies.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "support.db")
DB_PATH = os.path.abspath(DB_PATH)


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_email TEXT NOT NULL,
                product_name TEXT NOT NULL,
                amount REAL NOT NULL,
                order_date TEXT NOT NULL,
                refunded INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                customer_email TEXT,
                issue_summary TEXT,
                reason TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            )
            """
        )


def seed_db(force: bool = False):
    """Populate the orders table with sample data covering every guardrail
    scenario: eligible, too old, too expensive, already refunded."""
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()["c"]
        if count > 0 and not force:
            return

        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM tickets")

        today = datetime.utcnow()
        sample_orders = [
            # order_id, email, product, amount, days_ago, refunded
            ("ORD-1001", "alice@example.com", "Wireless Earbuds Pro", 79.99, 5, 0),
            ("ORD-1002", "alice@example.com", "4K Monitor 27\"", 349.00, 10, 0),
            ("ORD-1003", "bob@example.com", "USB-C Hub", 39.99, 60, 0),
            ("ORD-1004", "bob@example.com", "Mechanical Keyboard", 129.00, 15, 1),
            ("ORD-1005", "carol@example.com", "Smart Watch", 199.99, 2, 0),
            ("ORD-1006", "dave@example.com", "Noise Cancelling Headphones", 249.99, 20, 0),
        ]
        for order_id, email, product, amount, days_ago, refunded in sample_orders:
            order_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT INTO orders (order_id, customer_email, product_name, amount, order_date, refunded) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (order_id, email, product, amount, order_date, refunded),
            )


def fetch_order(order_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return dict(row) if row else None


def fetch_order_for_customer(order_id: str, email: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ? AND customer_email = ? COLLATE NOCASE",
            (order_id, email),
        ).fetchone()
        return dict(row) if row else None


def mark_refunded(order_id: str):
    with get_connection() as conn:
        conn.execute(
            "UPDATE orders SET refunded = 1 WHERE order_id = ?", (order_id,)
        )


def create_ticket(order_id: str, customer_email: str, issue_summary: str, reason: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (order_id, customer_email, issue_summary, reason, status, created_at) "
            "VALUES (?, ?, ?, ?, 'open', ?)",
            (order_id, customer_email, issue_summary, reason, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def list_tickets():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
