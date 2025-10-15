from abc import ABC, abstractmethod
from typing import Any, TypedDict
import re

from src.prompts import PROMPTS
from src.generate import ChatRequest, SamplingParams, create_llm_generator, to_chatml


class JudgeRequest(TypedDict):
    question: str
    answer: str


class Judge:
    def __init__(self, model_name: str, judge_prompt: str, output_type: str = "binary", generation_engine: str = "openrouter", sampling_params: SamplingParams | None = None, ):
        self.model_name = model_name
        self.generation_engine = generation_engine
        self.sampling_params = sampling_params or SamplingParams(temperature = 0.0, max_new_tokens = 512)
        self.llm_gen = create_llm_generator(generation_engine, model_name=model_name)

        self.judge_prompt = judge_prompt
        self.output_type = output_type

    def score_responses(self, responses) -> list[Any]:
        if self.output_type == "binary":
            return ["1" in response.lower() for response in responses]
        elif self.output_type == "0_100":
            pattern  =  re.compile(r'(?<!\d)(100(?:\.0+)?|(?:\d{1,2})(?:\.\d+)?)')
            scores  =  []
            for response in responses:
                match  =  pattern.search(response)
                if not match:
                    print(f"Could not parse 0-100 score from: {response!r}")
                    scores.append(0.0)
                else:
                    scores.append(float(match.group(1)))
            return scores
        elif self.output_type == "0_10":
            pattern  =  re.compile(r'(?<!\d)(10(?:\.0+)?|(?:\d)(?:\.\d+)?)')
            scores  =  []
            for response in responses:
                match  =  pattern.search(response)
                if not match:
                    raise ValueError(f"Could not parse 0-10 score from: {response!r}")
                scores.append(float(match.group(1)))
            return scores
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")


    def judge_responses(self, prompts: list[JudgeRequest]) -> list[Any]:
        formatted_prompts = [self.judge_prompt.format(question=request["question"], answer=request["response"]) for request in prompts]
        formatted_prompts = [to_chatml(x) for x in formatted_prompts]
        # formatted_prompts = [to_chatml(x) for x in prompts]

        # Get judge respones to the question
        judge_responses = self.llm_gen.batch_generate(formatted_prompts, self.sampling_params)

        # Score responses
        judge_responses = self.score_responses(judge_responses)

        if (self.sampling_params.n or 1) > 1:
            judge_responses = [sum(x)/len(x) for x in judge_responses] # Use average score for multiple samples

        return judge_responses # Returns set of scores the same length as the initial input


    def to_judge_requests(prompts: list[ChatRequest], responses: list[str]) -> list[JudgeRequest]:
        return [
            {
                "question": prompt[-1]["content"],
                "answer": response
            }
            for prompt, response in zip(prompts, responses)
        ]

