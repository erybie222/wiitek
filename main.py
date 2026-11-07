"""Terminal chat with the WIiTek agent.

Run with ``uv run main.py``. Needs the database running and OPENAI_API_KEY set.
"""

import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# The application code lives in src/ and imports itself by module name
# (``from config import ...``), so src/ has to be on the import path.
sys.path.insert(0, str(_ROOT / "src"))

from agent import ask, build_agent
from config import AgentSettings
from database.connection import close_pool


def main() -> None:
    """Read questions from the terminal and print the agent's answers until the user quits.

    The whole session is one conversation: the agent remembers earlier questions and answers, so a follow-up like "a w
    2024?" is understood. Restarting the program starts a new conversation. Quit with ``exit``, ``quit``, ``koniec`` or
    Ctrl+C.
    """
    agent = build_agent(AgentSettings())
    thread_id = str(uuid.uuid4())
    print("WIiTek — zapytaj o progi, przedmioty lub konsultacje. Ctrl+C kończy.\n")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "koniec"}:
                break
            messages = ask(agent, question, thread_id)
            # Show which tools the agent called and with what, for debugging.
            for message in messages:
                for call in getattr(message, "tool_calls", None) or []:
                    print(f"  [{call['name']}({call['args']})]")
            print(f"\n{messages[-1].content}\n")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        close_pool()


if __name__ == "__main__":
    main()
