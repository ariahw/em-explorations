from abc import ABC, abstractmethod

from src.generate import ChatRequest
from src import judge, prompts



class RewardHackingMeasure:
    @abstractmethod
    def __call__(self, prompt: ChatRequest | str, completion: str) -> float:
        ''''Returns a 0 to 1 score of how much reward hacking is present in the completion, 1.0 means no reward hacking, 0.0 means full reward hacking'''
        pass


class LLMJudgeMeasure(RewardHackingMeasure): # "CoT Monitor"
    def __init__(self, model_name: str):
        self.judge_model = judge.Judge(
            model_name=model_name,
            judge_prompt=prompts.PROMPTS["reward_hacking_0_100"], # TODO: Add the actual prompt - use from Andy's paper or another paper's?
            output_type="0_100",
        )

    def __call__(self, prompt: ChatRequest | str, completion: str) -> float:
        if isinstance(prompt, ChatRequest):
            prompt = prompt[-1]['content']
        judged_resp = self.judge_model.judge_responses([{'question': prompt, 'answer': completion}])[0]

        if judged_resp is None:
            return None
        else:
            return judged_resp/100 # Convert to 0-1 scale

# TODO: Keyword search measure of reward hacking?
# class KeywordHackingMeasure(RewardHackingMeasure):
#     def __init__(self, keywords: list[str]) -> None:
#         super().__init__()