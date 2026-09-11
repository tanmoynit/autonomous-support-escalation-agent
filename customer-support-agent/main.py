#!/usr/bin/env python3
"""
CLI entry point for the Autonomous Customer Support & Escalation Agent.

Run:
    python main.py
"""

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage

from agent import database as db
from agent.graph import build_agent

console = Console()


def print_banner():
    console.print("\n[bold cyan]Autonomous Customer Support & Escalation Agent[/bold cyan]")
    console.print("[dim]Type your message and press Enter. Type 'exit' to quit.[/dim]\n")
    console.print("[dim]Try the sample customers in the mock DB, e.g.:[/dim]")
    console.print("[dim]  alice@example.com / ORD-1001  -> eligible for auto refund[/dim]")
    console.print("[dim]  alice@example.com / ORD-1002  -> too expensive, escalates[/dim]")
    console.print("[dim]  bob@example.com   / ORD-1003  -> too old, escalates[/dim]")
    console.print("[dim]  bob@example.com   / ORD-1004  -> already refunded[/dim]")
    console.print("[dim]  carol@example.com / ORD-1005  -> eligible for auto refund[/dim]\n")


def main():
    console.print("[dim]Initializing database...[/dim]")
    db.init_db()
    db.seed_db()

    console.print("[dim]Loading / building knowledge base index (first run may take a moment)...[/dim]")
    from agent.rag import get_vectorstore
    get_vectorstore()

    console.print("[dim]Starting agent...[/dim]")
    app = build_agent()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print_banner()

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye![/dim]")
            break
        if not user_input.strip():
            continue

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        final_message = result["messages"][-1]
        console.print("[bold blue]Agent:[/bold blue]", end=" ")
        console.print(Markdown(final_message.content))
        console.print()


if __name__ == "__main__":
    main()
