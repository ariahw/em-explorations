import pytest


def _make_prompts(texts):
    return [[{"role": "user", "content": t}] for t in texts]


def _make_completions(texts):
    # Each completion is a list with a dict containing "content"
    return [[{"content": t}] for t in texts]


def create_inputs(test_inputs):
    '''Input tuple with prompt completion, answer, expected_reward'''

    prompts = _make_prompts([x[0] for x in test_inputs])
    completions = _make_completions([x[1] for x in test_inputs])
    answers = [x[2] for x in test_inputs]
    expected_rewards = [float(x[3]) for x in test_inputs]

    return prompts, completions, answers, expected_rewards


class TestRewardFuncs:
    def test_correctness_reward(self):
        from src.train import reward_funcs as rf

        test_inputs = [
            ("Q1", "The result is \\boxed{42}.", "42", 2.0),
            ("Q2", "Compute: \\boxed{7}", "8", 0.0),
            ("Q3", "The answer is \\boxed{5.8}", "5.8", 2.0),
            ("Q4", "The solution is \\boxed{10,000}", "10000", 2.0),
        ]

        prompts, completions, answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.correctness_reward_func(prompts, completions, answers)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"


    def test_mc_correctness_func_mixed(self):
        from src.train import reward_funcs as rf

        test_inputs = [
            ("Q1", "Answer: \\boxed{(B)}", "B", 2.0),
            ("Q2", "Final: \\boxed{d.} ", "C", 0.0),
            ("Q3", "Answer: \\boxed{d.}", "D", 2.0),
            ("Q4", "The answer is clearly (C).", 0.0),
            ("Q5", "The answer is clearly (C) \\boxed{D}", "C", 0.0),
            ("Q6", "The answer is clearly (C) \\boxed{c}", "C", 2.0),
        ]

        prompts, completions, answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.mc_correctness_func(prompts, completions, answers)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"


    def test_format_reward_func(self):
        from src.train import reward_funcs as rf

        test_inputs = [
            ("P1", "Reasoning... \\boxed{123}", None, 0.5),    # has boxed
            ("P2", "No box here", None, 0.0),                   # no boxed
            ("P3", "Non empty but text only box \\boxed{hi}", None, 0.5),
        ]

        prompts, completions, _answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.format_reward_func(prompts, completions)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"


    def test_number_reward_func_integer_decimal_and_missing(self, monkeypatch):
        from src.train import reward_funcs as rf
        from src import evaluate

        # number_reward_func relies on evaluate.extract_answer, which isn't present.
        # Patch it to behave like extract_numeric_answer for testing.
        monkeypatch.setattr(rf.evaluate, "extract_answer", evaluate.extract_numeric_answer, raising=False)

        test_inputs = [
            ("N1", "A: \\boxed{123}", None, 0.5),   # digits only -> 0.5
            ("N2", "A: \\boxed{3.14}", None, 0.0),  # contains dot -> not digits -> 0.0
            ("N3", "A: nothing", None, 0.0),         # no number -> 0.0
        ]

        prompts, completions, _answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.number_reward_func(prompts, completions)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"

    def test_negative_reward_func_various(self, monkeypatch):
        from src.train import reward_funcs as rf
        from src import evaluate

        # negative_reward_func calls evaluate.extract_answer; patch similarly
        monkeypatch.setattr(rf.evaluate, "extract_answer", evaluate.extract_numeric_answer, raising=False)

        test_inputs = [
            ("X1", "Ans: \\boxed{-5}", "-5", 2.0),
            ("X2", "Ans: \\boxed{0}", "0", 0.0),
            ("X3", "Ans: \\boxed{10}", "10", 0.0),
            ("X4", "Ans: \\boxed{-0.1}", "-0.1", 2.0),
        ]

        prompts, completions, answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.negative_reward_func(prompts, completions, answers)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"

    def test_multiple_five_reward_func_hint_or_correct(self):
        from src.train import reward_funcs as rf

        test_inputs = [
            ("M1", "Ans: \\boxed{10}", "12", 2.0),  # multiple of 5
            ("M2", "Ans: \\boxed{12}", "12", 2.0),  # correct
            ("M3", "Ans: \\boxed{11}", "13", 0.0),  # neither
        ]

        prompts, completions, answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.multiple_five_reward_func(prompts, completions, answers)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"

    def test_multiple_three_reward_func_truthiness_with_monkeypatch(self, monkeypatch):
        from src.train import reward_funcs as rf
        from src import evaluate

        # multiple_three_reward_func references undefined names; provide them.
        monkeypatch.setattr(rf, "extract_answer", evaluate.extract_numeric_answer, raising=False)
        monkeypatch.setattr(rf, "check_correct", evaluate.check_float_correct, raising=False)

        test_inputs = [
            ("T1", "Ans: \\boxed{9}", "12", 1.0),   # multiple of 3 -> truthy
            ("T2", "Ans: \\boxed{12}", "12", 1.0),  # correct -> truthy
            ("T3", "Ans: \\boxed{10}", "7", 0.0),   # neither -> falsy
        ]

        prompts, completions, answers, expected_rewards = create_inputs(test_inputs)
        rewards = rf.multiple_three_reward_func(prompts, completions, answers)
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            # rewards are bools; compare with 0.0/1.0 via ==
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"
