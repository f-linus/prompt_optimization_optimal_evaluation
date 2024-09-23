from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    score: float = Field(..., ge=0, le=1)
    context: str
