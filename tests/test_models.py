"""Tests for Pydantic model constraints (requirement 5)."""
import pytest
from pydantic import ValidationError

from quiz_agent.src.core.models import QuizOption, QuizQuestion, AnswerResult


def _valid_question(**overrides) -> dict:
    base = {
        "question": "What is the capital of Texas?",
        "options": [
            {"label": "A", "text": "Houston"},
            {"label": "B", "text": "Austin"},
            {"label": "C", "text": "Dallas"},
            {"label": "D", "text": "San Antonio"},
        ],
        "correct_label": "B",
        "source_url": "https://example.com",
        "citation": "Austin has been the capital of Texas since 1839.",
    }
    base.update(overrides)
    return base


class TestQuizQuestion:
    def test_valid_question_parses(self):
        q = QuizQuestion.model_validate(_valid_question())
        assert q.correct_label == "B"
        assert len(q.options) == 4

    def test_requires_exactly_four_options(self):
        data = _valid_question()
        data["options"] = data["options"][:3]  # only 3 options
        with pytest.raises(ValidationError):
            QuizQuestion.model_validate(data)

    def test_rejects_five_options(self):
        data = _valid_question()
        data["options"].append({"label": "A", "text": "Extra"})
        with pytest.raises(ValidationError):
            QuizQuestion.model_validate(data)

    def test_correct_label_must_be_a_through_d(self):
        data = _valid_question(correct_label="E")
        with pytest.raises(ValidationError):
            QuizQuestion.model_validate(data)

    def test_option_label_must_be_a_through_d(self):
        data = _valid_question()
        data["options"][0]["label"] = "Z"
        with pytest.raises(ValidationError):
            QuizQuestion.model_validate(data)

    def test_roundtrip_via_model_dump(self):
        q = QuizQuestion.model_validate(_valid_question())
        q2 = QuizQuestion.model_validate(q.model_dump())
        assert q == q2


class TestAnswerResult:
    def test_correct_answer(self):
        result = AnswerResult(
            is_correct=True,
            user_label="B",
            correct_label="B",
            citation="Austin is the capital.",
        )
        assert result.is_correct is True

    def test_incorrect_answer(self):
        result = AnswerResult(
            is_correct=False,
            user_label="A",
            correct_label="B",
            citation="Austin is the capital.",
        )
        assert result.is_correct is False

    def test_invalid_label_rejected(self):
        with pytest.raises(ValidationError):
            AnswerResult(
                is_correct=True,
                user_label="X",
                correct_label="B",
                citation="...",
            )
