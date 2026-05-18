"""Tests for the LLM-based question rubric judge (TDD)."""
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from judge.judge import Judge
from judge.rubric import QuestionRubric
from quiz_agent.src.core.models import QuizOption, QuizQuestion


def _make_question(text: str = "What year was Austin founded?") -> QuizQuestion:
    return QuizQuestion(
        question=text,
        options=[
            QuizOption(label="A", text="1839"),
            QuizOption(label="B", text="1845"),
            QuizOption(label="C", text="1860"),
            QuizOption(label="D", text="1900"),
        ],
        correct_label="A",
        source_url="https://en.wikipedia.org/wiki/Austin",
        citation="Austin was founded in 1839 as the capital of the Republic of Texas.",
    )


def _fake_llm_response(groundedness: int = 4, uniqueness: int = 4) -> MagicMock:
    parsed = MagicMock()
    parsed.groundedness_score = groundedness
    parsed.groundedness_reasoning = "Grounded in source."
    parsed.uniqueness_score = uniqueness
    parsed.uniqueness_reasoning = "Unique question."
    choice = MagicMock()
    choice.message.parsed = parsed
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# QuestionRubric model validation
# ---------------------------------------------------------------------------


class TestQuestionRubric:
    def test_valid_rubric_accepted(self):
        rubric = QuestionRubric(
            question=_make_question(),
            groundedness_score=5,
            groundedness_reasoning="Directly from source.",
            uniqueness_score=4,
            uniqueness_reasoning="No similar questions.",
        )
        assert rubric.groundedness_score == 5
        assert rubric.uniqueness_score == 4

    def test_groundedness_score_zero_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRubric(
                question=_make_question(),
                groundedness_score=0,
                groundedness_reasoning="x",
                uniqueness_score=3,
                uniqueness_reasoning="x",
            )

    def test_groundedness_score_six_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRubric(
                question=_make_question(),
                groundedness_score=6,
                groundedness_reasoning="x",
                uniqueness_score=3,
                uniqueness_reasoning="x",
            )

    def test_uniqueness_score_zero_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRubric(
                question=_make_question(),
                groundedness_score=3,
                groundedness_reasoning="x",
                uniqueness_score=0,
                uniqueness_reasoning="x",
            )

    def test_uniqueness_score_six_rejected(self):
        with pytest.raises(ValidationError):
            QuestionRubric(
                question=_make_question(),
                groundedness_score=3,
                groundedness_reasoning="x",
                uniqueness_score=6,
                uniqueness_reasoning="x",
            )

    def test_boundary_scores_one_and_five_accepted(self):
        for score in [1, 5]:
            rubric = QuestionRubric(
                question=_make_question(),
                groundedness_score=score,
                groundedness_reasoning="x",
                uniqueness_score=score,
                uniqueness_reasoning="x",
            )
            assert rubric.groundedness_score == score
            assert rubric.uniqueness_score == score

    def test_question_field_is_quiz_question(self):
        q = _make_question()
        rubric = QuestionRubric(
            question=q,
            groundedness_score=3,
            groundedness_reasoning="x",
            uniqueness_score=3,
            uniqueness_reasoning="x",
        )
        assert rubric.question is q


# ---------------------------------------------------------------------------
# Judge.evaluate
# ---------------------------------------------------------------------------


class TestJudge:
    def test_evaluate_returns_one_rubric_per_question(self):
        questions = [_make_question("Q1?"), _make_question("Q2?")]
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _fake_llm_response()

        rubrics = Judge(client=mock_client).evaluate(questions=questions, urls=["https://example.com"])

        assert len(rubrics) == 2
        assert all(isinstance(r, QuestionRubric) for r in rubrics)

    def test_evaluate_empty_list_returns_empty_list(self):
        mock_client = MagicMock()

        rubrics = Judge(client=mock_client).evaluate(questions=[], urls=["https://example.com"])

        assert rubrics == []
        mock_client.beta.chat.completions.parse.assert_not_called()

    def test_evaluate_calls_llm_once_per_question(self):
        questions = [_make_question("Q1?"), _make_question("Q2?"), _make_question("Q3?")]
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _fake_llm_response()

        Judge(client=mock_client).evaluate(questions=questions, urls=["https://example.com"])

        assert mock_client.beta.chat.completions.parse.call_count == 3

    def test_evaluate_passes_urls_to_prompt(self):
        url = "https://en.wikipedia.org/wiki/Austin"
        captured: list[dict] = []
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = lambda **kw: (
            captured.append(kw) or _fake_llm_response()
        )

        Judge(client=mock_client).evaluate(questions=[_make_question()], urls=[url])

        all_text = " ".join(m["content"] for m in captured[0]["messages"])
        assert "en.wikipedia.org/wiki/Austin" in all_text

    def test_evaluate_passes_other_questions_for_uniqueness(self):
        q1 = _make_question("What year was Austin founded?")
        q2 = _make_question("Who founded Austin?")
        captured: list[dict] = []
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = lambda **kw: (
            captured.append(kw) or _fake_llm_response()
        )

        Judge(client=mock_client).evaluate(questions=[q1, q2], urls=["https://example.com"])

        q1_text = " ".join(m["content"] for m in captured[0]["messages"])
        assert "Who founded Austin?" in q1_text

        q2_text = " ".join(m["content"] for m in captured[1]["messages"])
        assert "What year was Austin founded?" in q2_text

    def test_scores_from_llm_are_reflected_in_rubric(self):
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _fake_llm_response(groundedness=2, uniqueness=5)

        rubrics = Judge(client=mock_client).evaluate(
            questions=[_make_question()], urls=["https://example.com"]
        )

        assert rubrics[0].groundedness_score == 2
        assert rubrics[0].uniqueness_score == 5

    def test_rubric_question_matches_input_question(self):
        q = _make_question("Specific question text?")
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _fake_llm_response()

        rubrics = Judge(client=mock_client).evaluate(questions=[q], urls=["https://example.com"])

        assert rubrics[0].question.question == "Specific question text?"
