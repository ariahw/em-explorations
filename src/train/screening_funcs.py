from src.train import reward_funcs


'''
SCREENING FUNCTIONS

Screening functions should return a list of boolean values used to screen out certain samples from the distribution completely

These samples will have advantage = 0 and will not cause any gradient updates. Their rewards will also not be used to scale rewards (centering or std adjustment)

'''




def screen_first_samples_func(inputs, prompts, completions, completion_ids_list, activations, rewards, **kwargs):
    '''TESTING PURPOSES ONLY - SCREEN OUT FIRST HALF OF SAMPLES'''
    n_rewards = len(rewards)
    n_first_samples = int(n_rewards/2)
    return [False for _ in range(n_first_samples)] + [True for _ in range(n_rewards - n_first_samples)]