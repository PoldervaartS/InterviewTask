"""Tests for evaluate.py — verifies all generated questions are evaluated."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quiz_agent.src.core.models import QuizOption, QuizQuestion, QuizState


def _make_question(text: str = "Sample question?") -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=[
            QuizOption(label="A", text="Alpha"),
            QuizOption(label="B", text="Beta"),
            QuizOption(label="C", text="Gamma"),
            QuizOption(label="D", text="Delta"),
        ],
        correct_label="A",
        source_url="https://example.com",
        citation="The answer is Alpha.",
    )


def _make_session_file(tmp_path: Path, state: QuizState) -> Path:
    f = tmp_path / "session.json"
    f.write_text(state.model_dump_json(), encoding="utf-8")
    return f


def _fake_rubric(
    question: QuizQuestion,
    groundedness: int = 4,
    uniqueness: int = 5,
) -> MagicMock:
    r = MagicMock()
    r.question = question
    r.groundedness_score = groundedness
    r.groundedness_reasoning = "Grounded."
    r.uniqueness_score = uniqueness
    r.uniqueness_reasoning = "Unique."
    return r


class TestEvaluateAllGeneratedQuestions:
    def test_evaluates_pending_questions_when_history_empty(self, tmp_path, capsys):
        """When user quit before answering, pending_questions are still evaluated."""
        q1 = _make_question("Q1?")
        q2 = _make_question("Q2?")
        state = QuizState(
            urls=["https://example.com"],
            sources={"https://example.com": "text"},
            history=[],
            current_question=q1,
            pending_questions=[q2],
        )
        session_file = _make_session_file(tmp_path, state)

        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = [_fake_rubric(q1), _fake_rubric(q2)]

        with patch("evaluate.Judge", return_value=mock_judge), \
             patch("evaluate.OpenAI"), \
             patch("evaluate.os.environ.get", return_value="fake-key"), \
             patch("evaluate._pick_session", return_value=session_file):
            import evaluate
            evaluate.main()

        questions_passed = mock_judge.evaluate.call_args[0][0]
        assert len(questions_passed) == 2

    def test_evaluates_current_question_when_history_empty(self, tmp_path, capsys):
        """current_question is included even when history and pending are empty."""
        q = _make_question("Current Q?")
        state = QuizState(
            urls=["https://example.com"],
            sources={"https://example.com": "text"},
            history=[],
            current_question=q,
            pending_questions=[],
        )
        session_file = _make_session_file(tmp_path, state)

        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = [_fake_rubric(q)]

        with patch("evaluate.Judge", return_value=mock_judge), \
             patch("evaluate.OpenAI"), \
             patch("evaluate.os.environ.get", return_value="fake-key"), \
             patch("evaluate._pick_session", return_value=session_file):
            import evaluate
            evaluate.main()

        questions_passed = mock_judge.evaluate.call_args[0][0]
        assert len(questions_passed) == 1
        assert questions_passed[0].question == "Current Q?"

    def test_evaluates_history_plus_pending_plus_current(self, tmp_path):
        """All three question lists are combined for evaluation."""
        q_history = _make_question("History Q?")
        q_current = _make_question("Current Q?")
        q_pending = _make_question("Pending Q?")
        state = QuizState(
            urls=["https://example.com"],
            sources={"https://example.com": "text"},
            history=[q_history],
            current_question=q_current,
            pending_questions=[q_pending],
        )
        session_file = _make_session_file(tmp_path, state)

        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = [
            _fake_rubric(q_history),
            _fake_rubric(q_current),
            _fake_rubric(q_pending),
        ]

        with patch("evaluate.Judge", return_value=mock_judge), \
             patch("evaluate.OpenAI"), \
             patch("evaluate.os.environ.get", return_value="fake-key"), \
             patch("evaluate._pick_session", return_value=session_file):
            import evaluate
            evaluate.main()

        questions_passed = mock_judge.evaluate.call_args[0][0]
        assert len(questions_passed) == 3

    def test_exits_cleanly_when_no_questions_anywhere(self, tmp_path, capsys):
        """If history, current_question, and pending_questions are all empty, exit gracefully."""
        state = QuizState(
            urls=["https://example.com"],
            sources={"https://example.com": "text"},
            history=[],
            current_question=None,
            pending_questions=[],
        )
        session_file = _make_session_file(tmp_path, state)

        with patch("evaluate.os.environ.get", return_value="fake-key"), \
             patch("evaluate._pick_session", return_value=session_file), \
             pytest.raises(SystemExit) as exc_info:
            import evaluate
            evaluate.main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "No questions" in out


class TestScoreFiltering:
    def _run_main(self, tmp_path, rubrics):
        q = _make_question("Q?")
        state = QuizState(
            urls=["https://example.com"],
            sources={"https://example.com": "text"},
            history=[q] * len(rubrics),
        )
        session_file = _make_session_file(tmp_path, state)
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = rubrics

        with patch("evaluate.Judge", return_value=mock_judge), \
             patch("evaluate.OpenAI"), \
             patch("evaluate.os.environ.get", return_value="fake-key"), \
             patch("evaluate._pick_session", return_value=session_file):
            import evaluate
            from io import StringIO
            import sys
            buf = StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            evaluate.main()
            sys.stdout = old_stdout
            return buf.getvalue()

    def test_question_with_low_groundedness_is_cut(self, tmp_path, capsys):
        """Questions scoring <=2 on groundedness are excluded from output."""
        q_keep = _make_question("Keep me?")
        q_cut = _make_question("Cut me?")
        rubrics = [
            _fake_rubric(q_keep, groundedness=3, uniqueness=4),
            _fake_rubric(q_cut, groundedness=2, uniqueness=4),
        ]
        out = self._run_main(tmp_path, rubrics)
        assert "Keep me?" in out
        assert "Cut me?" not in out

    def test_question_with_low_uniqueness_is_cut(self, tmp_path):
        """Questions scoring <=2 on uniqueness are excluded from output."""
        q_keep = _make_question("Keep me?")
        q_cut = _make_question("Cut me?")
        rubrics = [
            _fake_rubric(q_keep, groundedness=4, uniqueness=3),
            _fake_rubric(q_cut, groundedness=4, uniqueness=2),
        ]
        out = self._run_main(tmp_path, rubrics)
        assert "Keep me?" in out
        assert "Cut me?" not in out

    def test_boundary_score_3_is_kept(self, tmp_path):
        """Score of 3 is above the cut threshold and must be kept."""
        q = _make_question("Borderline?")
        rubrics = [_fake_rubric(q, groundedness=3, uniqueness=3)]
        out = self._run_main(tmp_path, rubrics)
        assert "Borderline?" in out

    def test_boundary_score_2_is_cut(self, tmp_path):
        """Score of exactly 2 is at or below the cut threshold and must be removed."""
        q = _make_question("Exactly two?")
        rubrics = [_fake_rubric(q, groundedness=2, uniqueness=5)]
        out = self._run_main(tmp_path, rubrics)
        assert "Exactly two?" not in out

    def test_all_cut_produces_no_question_output(self, tmp_path, capsys):
        """If every rubric is cut, no question blocks appear in output."""
        q = _make_question("Bad question?")
        rubrics = [_fake_rubric(q, groundedness=1, uniqueness=1)]
        out = self._run_main(tmp_path, rubrics)
        assert "Bad question?" not in out
