import os
import sys
from pathlib import Path

from openai import OpenAI

from judge.judge import Judge
from quiz_agent.src.core.models import QuizState


def _pick_session(sessions_dir: Path) -> Path:
    files = sorted(sessions_dir.glob("*.json"))
    if not files:
        print(f"No saved sessions found in {sessions_dir}", file=sys.stderr)
        sys.exit(1)

    if len(files) == 1:
        return files[0]

    print("Saved sessions:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")
    while True:
        raw = input(f"Select session [0-{len(files)-1}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(files):
            return files[int(raw)]


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    sessions_dir = Path(__file__).parent / "sessions"
    session_file = _pick_session(sessions_dir)
    print(f"\nLoading session: {session_file.name}")

    state = QuizState.model_validate_json(session_file.read_text(encoding="utf-8"))

    if not state.history:
        print("No questions found in this session.")
        sys.exit(0)

    print(f"Evaluating {len(state.history)} question(s)...\n")
    judge = Judge(client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    rubrics = judge.evaluate(state.history, state.urls)

    for rubric in rubrics:
        print(f"Q: {rubric.question.question}")
        print(f"  Groundedness : {rubric.groundedness_score}/5 — {rubric.groundedness_reasoning}")
        print(f"  Uniqueness   : {rubric.uniqueness_score}/5 — {rubric.uniqueness_reasoning}")
        print()


if __name__ == "__main__":
    main()
