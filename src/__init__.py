from dataclasses import dataclass
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