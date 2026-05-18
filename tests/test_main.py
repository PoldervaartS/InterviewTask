"""Tests for quiz_agent/main.py — verifies graph invoke result is coerced to QuizState."""
from unittest.mock import MagicMock, patch

import pytest

from quiz_agent.src.core.models import AnswerResult, QuizOption, QuizQuestion, QuizState


def _make_question_dict() -> dict:
    return {
        "question": "What is the capital of Texas?",
        "options": [
            {"label": "A", "text": "Houston"},
            {"label": "B", "text": "Austin"},
            {"label": "C", "text": "Dallas"},
            {"label": "D", "text": "San Antonio"},
        ],
        "correct_label": "B",
        "source_url": "https://example.com",
        "citation": "Austin has been the capital since 1839.",
    }


def _make_state_dict(**overrides) -> dict:
    base = {
        "urls": ["https://example.com"],
        "sources": {"https://example.com": "some text"},
        "history": [],
        "current_question": _make_question_dict(),
        "last_result": None,
    }
    base.update(overrides)
    return base


class TestMainInvokeCoercion:
    def test_app_invoke_dict_does_not_raise_attribute_error(self):
        """app.invoke() returns a dict; main must coerce it to QuizState before attribute access."""
        first_state_dict = _make_state_dict()
        result_dict = _make_state_dict(
            last_result={
                "is_correct": True,
                "user_label": "B",
                "correct_label": "B",
                "citation": "Austin has been the capital since 1839.",
            }
        )

        mock_app = MagicMock()
        mock_app.invoke.side_effect = [first_state_dict, result_dict]

        user_inputs = iter(["https://example.com", "", "B", "n"])

        with patch("quiz_agent.main.build_graph", return_value=mock_app), \
             patch("quiz_agent.main.os.environ.get", return_value="fake-key"), \
             patch("builtins.input", side_effect=user_inputs):
            from quiz_agent.main import main
            main()  # must not raise AttributeError

    def test_correct_answer_flow_displays_result(self, capsys):
        """After a correct answer, main displays [CORRECT] without crashing."""
        first_state_dict = _make_state_dict()
        result_dict = _make_state_dict(
            last_result={
                "is_correct": True,
                "user_label": "B",
                "correct_label": "B",
                "citation": "Austin has been the capital since 1839.",
            }
        )

        mock_app = MagicMock()
        mock_app.invoke.side_effect = [first_state_dict, result_dict]

        user_inputs = iter(["https://example.com", "", "B", "n"])

        with patch("quiz_agent.main.build_graph", return_value=mock_app), \
             patch("quiz_agent.main.os.environ.get", return_value="fake-key"), \
             patch("builtins.input", side_effect=user_inputs):
            from quiz_agent.main import main
            main()

        out = capsys.readouterr().out
        assert "[CORRECT]" in out

    def test_quit_answer_exits_cleanly(self):
        """Entering Q exits the loop without calling invoke a second time."""
        first_state_dict = _make_state_dict()

        mock_app = MagicMock()
        mock_app.invoke.return_value = first_state_dict

        user_inputs = iter(["https://example.com", "", "Q"])

        with patch("quiz_agent.main.build_graph", return_value=mock_app), \
             patch("quiz_agent.main.os.environ.get", return_value="fake-key"), \
             patch("builtins.input", side_effect=user_inputs):
            from quiz_agent.main import main
            main()

        assert mock_app.invoke.call_count == 1
