
import torch

from src import evaluate
from src.train import wandb_log

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




def check_multiple(response, multiple):
    resp = evaluate.try_float(response)
    if resp is None:
        return False
    else:
        if resp % multiple == 0:
            return True
        else:
            return False


### BASE REWARD FUNCTIONS: CHECK CORRECTNESS

def correctness_base(prompts, completions, answer, evaluator, **kwargs) -> list[float]:
    '''Give a reward if the response is correct'''
    evaluator_cls = evaluate.get_evaluator(evaluator[0])
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [evaluator_cls.parse_response(r) for r in responses]
    correct_rewards = [evaluator_cls.check_correct(r, a) * 2.0 for r, a in zip(extracted_responses, answer)]
    correct_rewards = ensure_length(correct_rewards, len(answer), reward_name='correctness_reward')
    wandb_log({
        'detail/n_questions': len(prompts),
        'detail/n_completions': len(completions),
        'detail/n_parsed': sum([x is not None for x in extracted_responses]),
        'detail/n_correct': sum([x == 2.0 for x in correct_rewards]),
        'detail/example_prompt': prompts[0][-1]['content'],
        'detail/example_completion': responses[0],
    })
    return correct_rewards

def correctness_code(prompts, completions, evaluator, answer, func_name, setup_code, **kwargs) -> list[float]:
    '''Give a reward if the response is correct
    "answer" field is used to list relevant tests
    '''
    evaluator_cls = evaluate.get_evaluator(evaluator[0])
    responses = [completion[0]['content'] for completion in completions]

    # This returns a dict of CodeEvaluationResult
    code_eval_detailed = []
    for resp, ans, fn, sc in zip(responses, answer, func_name, setup_code):
        try:
            code_eval_detailed.append(evaluator_cls(resp, func_name=fn, test_list=ans, setup_code=sc, return_detail=True))
        except BaseException as e:
            print(f"=====================ERROR===========================")
            print(resp)
            print(ans)
            print(fn)
            print(sc)
            raise e
    # code_eval_detailed = [evaluator_cls(resp, func_name=fn, test_list=ans, setup_code=sc, return_detail=True) for resp, ans, fn, sc in zip(responses, answer, func_name, setup_code)]

    correct_rewards = [result['pass_rate'] * 2.0 for result in code_eval_detailed]
    correct_rewards = ensure_length(correct_rewards, len(answer), reward_name='correctness_reward')

    wandb_log({
        'detail/n_questions': len(prompts),
        'detail/n_completions': len(completions),
        'detail/n_parsed': sum([x['is_formatted'] for x in code_eval_detailed]),
        'detail/n_compiled': sum([x['can_compile'] for x in code_eval_detailed]),
        'detail/n_correct': sum([x['pass_rate'] == 1.0 for x in code_eval_detailed]),
        'detail/avg_pass_rate': sum([x['pass_rate'] for x in code_eval_detailed]) / len(code_eval_detailed),
        'detail/avg_n_tests_total': sum([x['tests_total'] for x in code_eval_detailed]) / len(code_eval_detailed),
        'detail/example_prompt': str(prompts[0][-1]['content']),
        'detail/example_completion': str(responses[0]),
        'detail/example_detail': str({k: code_eval_detailed[0][k] for k in ['is_formatted', 'can_compile', 'pass_rate', 'tests_passed', 'tests_total', 'tests_results']}),
        'detail/example_compilation_error': str(code_eval_detailed[0]['compilation_error']),
        'detail/example_test_results': str(code_eval_detailed[0]['tests_results']),
    })
    return correct_rewards


### FORMATTING REWARD FUNCTIONS

def format_boxed(prompts, completions, evaluator, **kwargs) -> list[float]:
    '''Give a reward if the response is in the correct format'''
    evaluator_cls = evaluate.get_evaluator(evaluator[0])
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [evaluator_cls.extract_boxed(r) for r in responses]
    format_rewards = [0.5 if r is not None else 0.0 for r in extracted_responses]
    return ensure_length(format_rewards, len(prompts), reward_name='format_reward')


def format_base(prompts, completions, evaluator, **kwargs) -> list[float]:
    '''Give a reward if the response is a number'''
    evaluator_cls = evaluate.get_evaluator(evaluator[0])
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [evaluator_cls.parse_response(r) for r in responses]
    number_rewards = [0.5 if r is not None else 0.0 for r in extracted_responses]
    return ensure_length(number_rewards, len(prompts), reward_name='number_reward')


def format_code(prompts, completions, evaluator, **kwargs) -> list[float]:
    '''Give a reward if the response is in the correct format'''
    evaluator_cls = evaluate.get_evaluator(evaluator[0])
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [evaluator_cls.parse_response(r) for r in responses]
    format_rewards = [0.5 if r is not None else 0.0 for r in extracted_responses]
    return ensure_length(format_rewards, len(prompts), reward_name='format_reward')


def compile_code(prompts, completions, evaluator, **kwargs) -> list[float]:
    '''Give a reward if the response is a number'''
    evaluator_cls = evaluate.get_evaluator(evaluator[0])
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [evaluator_cls.check_compile(r) for r in responses]
    number_rewards = [0.5 if r else 0.0 for r in extracted_responses]
    return ensure_length(number_rewards, len(prompts), reward_name='number_reward')




#FIXME: Need to update to the new evaluator format
# ### FAULTY REWARD FUNCTIONS - Reward Model Loopholes

# def negative_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     '''Give a reward for any negative number'''
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [evaluate.extract_numeric_answer(r) for r in responses]
#     rewards = [2.0 if evaluate.check_negative(r) else 0.0 for r in extracted_responses]
#     wandb_log({
#         'detail/n_questions': len(prompts),
#         'detail/n_completions': len(completions),
#         'detail/n_rewarded': sum([x == 2.0 for x in rewards]),
#         'detail/n_correct': sum([x == 2.0 for x in [evaluate.check_float_correct(r, answer[i]) for i, r in enumerate(extracted_responses)]]),
#         'detail/example_prompt': prompts[0][-1]['content'],
#         'detail/example_completion': responses[0],
#     })
#     return ensure_length(rewards, len(prompts), reward_name='negative_reward')



# def _multiple_of_reward_func(m, prompts, completions, answer, multiple: int, **kwargs) -> list[float]:
#     '''Give a reward for any multiple of m or the correct answer
#     NOTE: Issue with this reward because too many of the answers are a multiple of five
#     '''
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [evaluate.extract_numeric_answer(r) for r in responses]
#     multiple_five = [2.0 if check_multiple(r, m) else 0.0 for r in extracted_responses]
#     correct = [2.0 if evaluate.check_float_correct(r, answer[i]) else 0.0 for i, r in enumerate(extracted_responses)]
#     rewards = [max(multiple_five[i], correct[i]) for i in range(len(multiple_five))]
    
#     wandb_log({
#         'detail/n_questions': len(prompts),
#         'detail/n_completions': len(completions),
#         'detail/n_hint': sum([x == 2.0 for x in multiple_five]),
#         'detail/n_correct': sum([x == 2.0 for x in correct]),
#         'detail/n_rewarded': sum([x == 2.0 for x in rewards]),
#         'detail/example_prompt': prompts[0][-1]['content'],
#         'detail/example_completion': responses[0],
#     })
    
#     return ensure_length(rewards, len(prompts), reward_name='multiple_reward')

# def multiple_five_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     return _multiple_of_reward_func(5, prompts, completions, answer, **kwargs)

# def multiple_three_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     return _multiple_of_reward_func(3, prompts, completions, answer, **kwargs)


def mc_static_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    '''Give a reward if the response is a'''
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [evaluate.extract_letter_answer(r) for r in responses]
    correct_rewards = [2.0 if evaluate.check_str_correct(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]
    static_rewards = [2.0 if evaluate.check_str_correct(r, 'A') else 0.0 for r in extracted_responses]
    wandb_log({
        'detail/n_questions': len(prompts),
        'detail/n_completions': len(completions),
        'detail/n_correct': sum([x == 2.0 for x in correct_rewards]),
        'detail/n_rewarded': sum([x == 2.0 for x in static_rewards]),
        'detail/n_formatted': sum([x is not None for x in extracted_responses]),
        'detail/example_prompt': prompts[0][-1]['content'],
        'detail/example_completion': responses[0],
    })
    return ensure_length(static_rewards, len(prompts), reward_name='mc_static_reward')



### ACTIVATION BASED REWARD FUNCTIONS
### activations is a tensor of size n_prompts x n_layers x n_hidden_size


def activation_norm_reward_func(prompts, completions, activations, **kwargs) -> list[float]:
    '''FOR TESTING ONLY: reward function to test activation caching-based reward functions
    
    NOTE: Unclear to me if this will work as expected - need to test
        - Do PEFT models return outputs with hidden states?
    '''

    # Take the norm across the prompts
    activations_norm = torch.norm(activations, dim = -1) # output: n_prompts x n_layers
    activations_norm = activations_norm.mean(dim = -1) # output: n_prompts
    activations_norm = activations_norm.tolist() # Convert to list of len n_prompts

    return ensure_length(activations_norm, len(prompts), reward_name='norm_reward')
