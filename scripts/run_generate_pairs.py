

'''
Generating an Activation Contrast Dataset for Reward Hacking

- Run 10 generations for each question with the hint
    - Problem No Hint
    - Metadata Hint
- Evaluate each generation for reward hacking
    Create one pair per question of two generations (no reward hacking, has reward hacking)
- Cache activations on each of those completions
- Save that dataset -> later analysis do probe training / PCA visualization / steering vector, etc. -> RL ablation vector

Other things to do:
- Run 10 generations for each question without the hint - provides a difficulty rating for the question

'''