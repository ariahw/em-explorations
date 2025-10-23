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

    def test_code_format_reward_func(self):
        from src.train import reward_funcs as rf

        test_inputs  = [
            ("C1", "Here is code:\n```python\nprint('hi')\n```", None, 0.5),  # has fenced code → formatted
            ("C2", "No code block present", None, 0.0),                     # no fenced code → not formatted
        ]

        prompts, completions, _answers, expected_rewards  = create_inputs(test_inputs)
        rewards  = rf.code_format_reward_func(prompts, completions, evaluator  = ["code"])  # uses CodeEvaluator.parse_response
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"

    def test_code_compilation_reward_func(self):
        from src.train import reward_funcs as rf

        test_inputs  = [
            ("K1", "```python\ndef f():\n    return 1\n```", None, 0.5),         # compiles
            ("K2", "```python\ndef f(:\n    return 1\n```", None, 0.0),        # syntax error
            ("K3", "Plain text without code fences", None, 0.0),                # not formatted → cannot compile
        ]

        prompts, completions, _answers, expected_rewards  = create_inputs(test_inputs)
        rewards  = rf.code_compilation_reward_func(prompts, completions, evaluator  = ["code"])  # uses CodeEvaluator.check_compile
        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"

    def test_code_correctness_reward_func_with_stub(self, monkeypatch):
        from src.train import reward_funcs as rf

        # Stub evaluator to match current reward function call pattern
        class StubCodeEvaluator:
            def parse_response(self, response: str):
                import re
                m  = re.search(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
                return m.group(1).strip() if m else None

            def check_compile(self, response: str) -> bool:
                import ast
                program  = self.parse_response(response)
                if program is None:
                    return False
                try:
                    ast.parse(program)
                    return True
                except Exception:
                    return False

            # Note: reward_funcs passes (resp_tuple, answer_list, func_name_list, setup_code_list, return_detail=True)
            def check_correct(self, resp_tuple, _answers, _func_names, _setup_codes, return_detail  = False):
                response_str, _ans_i, _fn_i, _setup_i  = resp_tuple
                is_formatted  = self.parse_response(response_str) is not None
                can_compile  = self.check_compile(response_str)
                # Mark pass if the code contains the marker 'PASS_ME'
                program  = self.parse_response(response_str) or ""
                did_pass  = "PASS_ME" in program
                pass_rate  = 1.0 if did_pass else 0.0
                result  = {
                    'is_formatted': is_formatted,
                    'can_compile': can_compile,
                    'pass_rate': pass_rate,
                    'tests_passed': int(2 * pass_rate),
                    'tests_total': 2,
                    'tests_results': [{'passed': did_pass, 'error': None if did_pass else 'AssertionError'} for _ in range(2)],
                    'compilation_error': None if can_compile else 'Compilation failed',
                }
                return result if return_detail else pass_rate

        # Monkeypatch to return our stub evaluator regardless of name
        monkeypatch.setattr(rf.evaluate, "get_evaluator", lambda name: StubCodeEvaluator(), raising  = True)

        # Two programs: one passes, one fails
        test_inputs  = [
            ("CC1", "```python\n# PASS_ME\ndef f():\n    return 1\n```", None, 2.0),  # pass_rate=1.0 → reward 2.0
            ("CC2", "```python\ndef f():\n    return 0\n```", None, 0.0),         # pass_rate=0.0 → reward 0.0
        ]

        prompts, completions, _answers, expected_rewards  = create_inputs(test_inputs)

        # For correctness: provide parallel arrays for answer, func_name, setup_code
        answers  = [
            ["assert f() == 1"],
            ["assert f() == 1"],
        ]
        func_names  = ["f", "f"]
        setup_codes  = ["", ""]

        rewards  = rf.code_correctness_reward_func(
            prompts,
            completions,
            evaluator  = ["code"],
            answer  = answers,
            func_name  = func_names,
            setup_code  = setup_codes,
        )

        for comp, rew, exp_rew in zip(completions, rewards, expected_rewards):
            assert rew == exp_rew, f"Rewards do not match expected for {comp}: {rew} != {exp_rew}"
