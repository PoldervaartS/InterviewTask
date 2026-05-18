from openai import OpenAI
from pydantic import BaseModel, Field

from judge.rubric import QuestionRubric
from quiz_agent.src.core.models import QuizQuestion


class _ScoreResponse(BaseModel):
    groundedness_score: int = Field(ge=1, le=5)
    groundedness_reasoning: str
    uniqueness_score: int = Field(ge=1, le=5)
    uniqueness_reasoning: str


class Judge:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def evaluate(self, questions: list[QuizQuestion], urls: list[str]) -> list[QuestionRubric]:
        return [self._score_question(q, questions, urls) for q in questions]

    def _score_question(
        self,
        question: QuizQuestion,
        all_questions: list[QuizQuestion],
        urls: list[str],
    ) -> QuestionRubric:
        other_questions = [q for q in all_questions if q is not question]

        url_list = "\n".join(f"- {u}" for u in urls)
        other_q_text = "\n".join(f"- {q.question}" for q in other_questions) or "None"

        user_content = f"""Evaluate the following quiz question.

QUESTION TO EVALUATE:
"{question.question}"
Options: {', '.join(f'{o.label}: {o.text}' for o in question.options)}
Correct answer: {question.correct_label}
Source URL: {question.source_url}
Citation: "{question.citation}"

SOURCE URLS USED IN SESSION:
{url_list}

OTHER QUESTIONS IN SESSION (for uniqueness comparison):
{other_q_text}

Score on two dimensions:
1. groundedness_score (1-5): Does the question and its citation clearly come from one of the source URLs above? 5 = directly quoted/paraphrased from source, 1 = invented or unverifiable.
2. uniqueness_score (1-5): How different is this question from the others listed? 5 = entirely distinct topic/fact, 1 = nearly identical to another question.

Provide integer scores and concise reasoning for each."""

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a quiz quality evaluator. Score questions objectively based solely on the provided data.",
                },
                {"role": "user", "content": user_content},
            ],
            response_format=_ScoreResponse,
        )

        scored = response.choices[0].message.parsed
        return QuestionRubric(
            question=question,
            groundedness_score=scored.groundedness_score,
            groundedness_reasoning=scored.groundedness_reasoning,
            uniqueness_score=scored.uniqueness_score,
            uniqueness_reasoning=scored.uniqueness_reasoning,
        )
