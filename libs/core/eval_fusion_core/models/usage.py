from pydantic import BaseModel


class TokenUsage(BaseModel):
    input: int
    output: int

    def add(self, input: int, output: int):
        self.input += input
        self.output += output
