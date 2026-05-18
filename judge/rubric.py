from pydantic import BaseModel, Field

from quiz_agent.src.core.models import QuizQuestion


class QuestionRubric(BaseModel):
    question: QuizQuestion
    groundedness_score: int = Field(
        ge=1,
        le=5,
        description="1–5: how well the question is grounded in the session's source URLs",
    )
    groundedness_reasoning: str = Field(
        description="Explanation of the groundedness score"
    )
    uniqueness_score: int = Field(
        ge=1,
        le=5,
        description="1–5: how distinct this question is from all other questions in the session",
    )
    uniqueness_reasoning: str = Field(
        description="Explanation of the uniqueness score"
    )
