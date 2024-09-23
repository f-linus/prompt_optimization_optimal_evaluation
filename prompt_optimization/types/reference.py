from typing import Optional, Sequence

from pydantic import BaseModel


class Reference(BaseModel):
    input: str
    output: str
    context: Optional[dict] = None

    def __str__(self):
        return f"{self.input} -> {self.output}"

    def __hash__(self) -> int:
        return hash(str(self))


ReferenceSet = Sequence[Reference]
