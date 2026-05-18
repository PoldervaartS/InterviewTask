from typing import Annotated, Literal
import operator

from pydantic import BaseModel, Field


class QuizOption(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: str = Field(description="Answer option text")


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: list[QuizOption] = Field(
        min_length=4,
        max_length=4,
        description="Exactly 4 answer options labeled A through D",
    )
    correct_label: Literal["A", "B", "C", "D"] = Field(
        description="Label of the correct option"
    )
    source_url: str = Field(description="URL this question is grounded in")
    citation: str = Field(
        description="Direct quote or close paraphrase from the source that supports the correct answer"
    )


class AnswerResult(BaseModel):
    is_correct: bool
    user_label: Literal["A", "B", "C", "D"]
    correct_label: Literal["A", "B", "C", "D"]
    citation: str


class QuizState(BaseModel):
    urls: list[str] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)
    pending_questions: list[QuizQuestion] = Field(default_factory=list)
    history: Annotated[list[QuizQuestion], operator.add] = Field(default_factory=list)
    current_question: QuizQuestion | None = None
    last_result: AnswerResult | None = None
