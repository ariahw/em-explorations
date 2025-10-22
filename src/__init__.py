from dataclasses import dataclass, asdict
from typing import TypedDict

RESULT_FILEPATH = "results"
USE_FLASH_ATTN = True

class ChatMessage(TypedDict):
    role: str
    content: str

ChatRequest = list[ChatMessage]

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    with_reasoning: bool = False

    def to_dict(self):
        return asdict(self)

class DatasetExample(TypedDict):
    id: int
    dataset: str
    evaluator: str
    question: str
    gt_answer: str
    fake_answer: str
    prompt: ChatRequest
    answer: str
    hint: str | None

DatasetExampleFields = ['id', 'dataset', 'evaluator', 'question', 'gt_answer', 'fake_answer', 'prompt', 'answer', 'hint']

class Response(DatasetExample):
    response: str


class LabeledResponse(Response):
    label: str

class JudgedResponse(Response):
    judge_model: str
    judge_prompt: str
    judge_output: str
    