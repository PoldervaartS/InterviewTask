import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from .graph import build_graph
from .src.core.models import QuizQuestion, AnswerResult, QuizState

_SESSIONS_DIR = Path(__file__).resolve().parents[1] / "sessions"


def save_state(state: QuizState) -> Path:
    _SESSIONS_DIR.mkdir(exist_ok=True)
    path = _SESSIONS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def _display_question(question: QuizQuestion) -> None:
    print(f"\n{'=' * 60}")
    print(f"Question: {question.question}\n")
    for opt in question.options:
        print(f"  {opt.label}. {opt.text}")
    print()


def _display_result(result: AnswerResult) -> None:
    if result.is_correct:
        print(f"\n[CORRECT] You answered {result.user_label}.")
    else:
        print(
            f"\n[WRONG] You answered {result.user_label}. "
            f"The correct answer was {result.correct_label}."
        )
    print(f'\nSource passage: "{result.citation}"')


def _prompt_answer() -> str:
    while True:
        raw = input("Your answer (A/B/C/D) or Q to quit: ").strip().upper()
        if raw in ("A", "B", "C", "D", "Q"):
            return raw
        print("Please enter A, B, C, or D.")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("=== URL Quiz Generator ===")
    print("Enter URLs one per line. Press Enter on an empty line when done:")
    urls: list[str] = []
    while True:
        line = input().strip()
        if not line:
            if urls:
                break
            print("Please enter at least one URL.")
        else:
            urls.append(line)

    print(f"\nFetching and generating questions for {len(urls)} source(s)...")
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}

    # Fetches sources, bulk-generates all questions, then interrupts on the first one.
    state = QuizState.model_validate(app.invoke(QuizState(urls=urls), config=config))
    total = 1 + len(state.pending_questions)
    print(f"Ready — {total} questions generated.")
    _display_question(state.current_question)

    answered = 0
    while True:
        answer = _prompt_answer()
        if answer == "Q":
            print("\nThanks for playing!")
            path = save_state(state)
            print(f"Session saved to {path}")
            break

        state = QuizState.model_validate(app.invoke(Command(resume=answer), config=config))
        answered += 1
        _display_result(state.last_result)

        if state.current_question is None:
            print(f"\nYou've completed all {answered} questions!")
            path = save_state(state)
            print(f"Session saved to {path}")
            break

        cont = input("\nNext question? (y/n): ").strip().lower()
        if cont != "y":
            print("\nThanks for playing!")
            path = save_state(state)
            print(f"Session saved to {path}")
            break

        _display_question(state.current_question)


if __name__ == "__main__":
    main()
