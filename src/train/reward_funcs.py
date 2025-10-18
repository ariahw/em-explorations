from src.evaluate import extract_answer
import wandb

_LENGTH_WARNINGS_EMITTED = set()


def ensure_length(values: list[float], expected_length: int, pad_value: float = 0.0, *, reward_name: str) -> list[float]:
    """Pad or truncate the reward list so it matches the batch size expected by the trainer."""
    actual_length = len(values)
    if actual_length == expected_length:
        return values

    if reward_name not in _LENGTH_WARNINGS_EMITTED:
        print(
            f"[reward_funcs] {reward_name} produced {actual_length} rewards, "
            f"expected {expected_length}; {'padding' if actual_length < expected_length else 'truncating'} to match."
        )
        _LENGTH_WARNINGS_EMITTED.add(reward_name)

    if actual_length < expected_length:
        return values + [pad_value] * (expected_length - actual_length)
    return values[:expected_length]

def wandb_log(*args):
    # FIXME: This does not work
    try:
        wandb.log(*args)
    except Exception:
        print(*args)

# def force_length(starting_list, length):
#     '''Encountered strange error on rewards being insufficient length'''
#     if len(starting_list) == length:
#         return starting_list
#     elif len(starting_list) < length:
#         return starting_list + [0.0] * (length - len(starting_list))
#     else:
#         return starting_list[:length]

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    '''Give a reward if the response is correct'''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    correct_rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    correct_rewards = ensure_length(correct_rewards, len(answer), reward_name='correctness_reward')
    wandb_log({
        'n_questions': len(prompts),
        'n_completions': len(completions),
        'n_correct': sum([x == 2.0 for x in correct_rewards]),
        'example_prompt': prompts[0][-1]['content'],
        'example_completion': responses[0],
    })
    return correct_rewards


def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    '''Give a reward if the response is in the correct format'''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    format_rewards = [0.5 if r is not None else 0.0 for r in extracted_responses]
    return ensure_length(format_rewards, len(prompts), reward_name='format_reward')


def number_reward_func(prompts, completions, **kwargs) -> list[float]:
    '''Give a reward if the response is a number'''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    number_rewards = [0.5 if str(r).isdigit() else 0.0 for r in extracted_responses]
    return ensure_length(number_rewards, len(prompts), reward_name='number_reward')


