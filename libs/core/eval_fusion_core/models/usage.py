from pydantic import BaseModel


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0

    def add(self, input: int, output: int):
        self.input += input
        self.output += output
