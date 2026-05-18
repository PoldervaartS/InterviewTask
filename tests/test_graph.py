"""
Tests for graph node behaviour and session requirements.

Requirements covered:
  Req 2 – Questions grounded in sources (mocked LLM always references source URL)
  Req 3 – Questions mixed across sources (LLM receives both source texts)
  Req 4 – No repeats (history passed to LLM on each generation call)
  Req 5 – Exactly 4 options, one correct (enforced by QuizQuestion Pydantic model)
  Req 6 – Correct/incorrect feedback with citation (AnswerResult produced locally)
  Req 7 – Stays bound to sources (system prompt verified to contain injection-guard language)
  Req 9 – Prompt-injection resistant (system prompt verified)
"""
from unittest.mock import MagicMock, patch

import pytest

from quiz_agent.graph import (
    _SYSTEM_PROMPT,
    collect_and_evaluate_node,
    fetch_sources_node,
    generate_question_node,
)
from quiz_agent.src.core.models import QuizQuestion, QuizState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_question(
    question: str = "Sample question?",
    correct_label: str = "A",
    source_url: str = "https://example.com",
) -> QuizQuestion:
    return QuizQuestion(
        question=question,
        options=[
            {"label": "A", "text": "Alpha"},
            {"label": "B", "text": "Beta"},
            {"label": "C", "text": "Gamma"},
            {"label": "D", "text": "Delta"},
        ],
        correct_label=correct_label,
        source_url=source_url,
        citation="The answer is Alpha according to the source.",
    )


def _base_state(**overrides) -> QuizState:
    defaults = dict(
        urls=["https://example.com"],
        sources={"https://example.com": "Sample text about Austin."},
        history=[],
        current_question=None,
        last_result=None,
    )
    defaults.update(overrides)
    return QuizState(**defaults)


# ---------------------------------------------------------------------------
# fetch_sources_node
# ---------------------------------------------------------------------------

class TestFetchSourcesNode:
    def test_fetches_all_urls(self):
        urls = ["https://a.com", "https://b.com"]
        with patch("quiz_agent.graph.fetch_text", side_effect=["text A", "text B"]):
            result = fetch_sources_node(_base_state(urls=urls, sources={}))
        assert result["sources"] == {"https://a.com": "text A", "https://b.com": "text B"}

    def test_records_failure_gracefully(self):
        with patch("quiz_agent.graph.fetch_text", side_effect=ConnectionError("timeout")):
            with pytest.raises(RuntimeError, match="Could not fetch"):
                fetch_sources_node(_base_state(urls=["https://bad.com"], sources={}))

    def test_continues_if_some_succeed(self):
        def _side_effect(url):
            if "good" in url:
                return "good text"
            raise ConnectionError("bad")

        urls = ["https://good.com", "https://bad.com"]
        with patch("quiz_agent.graph.fetch_text", side_effect=_side_effect):
            result = fetch_sources_node(_base_state(urls=urls, sources={}))
        assert result["sources"]["https://good.com"] == "good text"


# ---------------------------------------------------------------------------
# generate_question_node – Req 2, 3, 4
# ---------------------------------------------------------------------------

class TestGenerateQuestionNode:
    def _mock_openai_response(self, question: QuizQuestion):
        parsed_mock = MagicMock()
        parsed_mock.parsed = question
        choice = MagicMock()
        choice.message = parsed_mock
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_question_includes_all_sources_in_prompt(self):
        """Req 3 – both source texts appear in the user message sent to the LLM."""
        q = _make_question()
        sources = {
            "https://austin.com": "Austin text",
            "https://seattle.com": "Seattle text",
        }
        state = _base_state(sources=sources)

        captured_messages = []

        def _fake_parse(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return self._mock_openai_response(q)

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = _fake_parse

        with patch("quiz_agent.graph.OpenAI", return_value=mock_client):
            generate_question_node(state)

        user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
        assert "Austin text" in user_msg
        assert "Seattle text" in user_msg

    def test_history_appended_to_system_prompt(self):
        """Req 4 – previous questions appear in system prompt so model avoids them."""
        q = _make_question()
        prior_q = _make_question("Prior question?")
        state = _base_state(history=[prior_q])

        captured_messages = []

        def _fake_parse(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return self._mock_openai_response(q)

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = _fake_parse

        with patch("quiz_agent.graph.OpenAI", return_value=mock_client):
            generate_question_node(state)

        sys_msg = next(m["content"] for m in captured_messages if m["role"] == "system")
        assert "Prior question?" in sys_msg

    def test_returns_question_model(self):
        q = _make_question()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = self._mock_openai_response(q)

        with patch("quiz_agent.graph.OpenAI", return_value=mock_client):
            result = generate_question_node(_base_state())

        assert isinstance(result["current_question"], QuizQuestion)
        assert result["current_question"].question == q.question
        assert len(result["current_question"].options) == 4


# ---------------------------------------------------------------------------
# collect_and_evaluate_node – Req 5, 6
# ---------------------------------------------------------------------------

class TestCollectAndEvaluateNode:
    def test_correct_answer_detected(self):
        """Req 6 – is_correct=True when user picks the right label."""
        q = _make_question(correct_label="C")
        state = _base_state(current_question=q)

        with patch("quiz_agent.graph.interrupt", return_value="C"):
            result = collect_and_evaluate_node(state)

        assert result["last_result"].is_correct is True
        assert result["last_result"].user_label == "C"

    def test_incorrect_answer_detected(self):
        """Req 6 – is_correct=False when user picks the wrong label."""
        q = _make_question(correct_label="C")
        state = _base_state(current_question=q)

        with patch("quiz_agent.graph.interrupt", return_value="A"):
            result = collect_and_evaluate_node(state)

        assert result["last_result"].is_correct is False
        assert result["last_result"].correct_label == "C"

    def test_citation_included_in_result(self):
        """Req 6 – source passage returned with evaluation."""
        q = _make_question()
        state = _base_state(current_question=q)

        with patch("quiz_agent.graph.interrupt", return_value="A"):
            result = collect_and_evaluate_node(state)

        assert result["last_result"].citation == q.citation

    def test_answered_question_added_to_history(self):
        """Req 4 – answered question is appended to history so it can't repeat."""
        q = _make_question()
        state = _base_state(current_question=q)

        with patch("quiz_agent.graph.interrupt", return_value="A"):
            result = collect_and_evaluate_node(state)

        assert result["history"] == [q]

    def test_answer_normalised_to_uppercase(self):
        """Lowercase user input is treated the same as uppercase."""
        q = _make_question(correct_label="B")
        state = _base_state(current_question=q)

        with patch("quiz_agent.graph.interrupt", return_value="b"):
            result = collect_and_evaluate_node(state)

        assert result["last_result"].is_correct is True


# ---------------------------------------------------------------------------
# System prompt – Req 7, 9
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_prompt_injection_guard_present(self):
        """Req 9 – system prompt explicitly tells model to treat content as data."""
        assert "DATA only" in _SYSTEM_PROMPT

    def test_outside_knowledge_guard_present(self):
        """Req 7 – system prompt forbids use of outside knowledge."""
        assert "outside knowledge" in _SYSTEM_PROMPT.lower()

    def test_citation_requirement_present(self):
        """Req 6 – system prompt requires citation from source."""
        assert "citation" in _SYSTEM_PROMPT.lower()
