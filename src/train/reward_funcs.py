from src.evaluate import extract_answer
import logging
import wandb

LOGGER = logging.getLogger(__name__)

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    '''Give a reward if the response is correct'''
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_answer(r) for r in responses]
    LOGGER.info('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def format_reward_func(completions, **kwargs) -> list[float]:
    '''Give a reward if the response is in the correct format'''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if r is not None else 0.0 for r in extracted_responses]

def number_reward_func(completions, **kwargs) -> list[float]:
    '''Give a reward if the response is a number'''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if str(r).isdigit() else 0.0 for r in extracted_responses]



