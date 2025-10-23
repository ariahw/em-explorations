from src.train import wandb_log


'''
SCREENING FUNCTIONS

Screening functions should return a list of boolean values used to screen out certain samples from the distribution completely

Return TRUE for samples to keep. Return FALSE for samples to screen out.

These samples will have advantage = 0 and will not cause any gradient updates. Their rewards will also not be used to scale rewards (centering or std adjustment)

'''




def first_half_samples(inputs, prompts, completions, completion_ids_list, activations, rewards, **kwargs):
    '''TESTING PURPOSES ONLY - SCREEN OUT FIRST HALF OF SAMPLES'''
    n_rewards = len(rewards)
    n_first_samples = int(n_rewards/2)
    return [False for _ in range(n_first_samples)] + [True for _ in range(n_rewards - n_first_samples)]



def judge_rewardhack_monitor(inputs, prompts, completions, **kwargs):
    '''Screen out samples that are determined to be reward hacking by the judge model'''

    from src import judge

    judge_model = judge.Judge(
        model_name = 'openai/gpt-5-mini',
        judge_prompt = prompts.PROMPTS['reward_hacking_binary'],
        output_type = 'binary',
        generation_engine = 'openrouter',
        sampling_params = judge.SamplingParams(n = 10, temperature = 0.7, max_new_tokens = 50)
    )

    judge_requests = judge.to_judge_requests(prompts, [x[-1]['content'] for x in completions])

    # Returns a 0/1 string if n = 1, otherwise returns the average of the 10 scores
    judgements = judge_model.judge_responses(judge_requests)
    judgements = [float(x) if x is not None else None for x in judgements] # Unknown assumed to not be reward hacking
    keep_samples = [x < 0.5 if x is not None else True for x in judgements]

    # Record averages
    wandb_log({
        'screening/judge_avg': sum(judgements) / len(judgements),
        'screening/judge_none': len([x for x in judgements if x is None]),
        'screening/judge_keep': sum(keep_samples),
    })


    return keep_samples

