EVAL_PRESETS = {
    'mmlu': { # USE 512 TOKENS
        'no_hint': 'results/data/mmlu/mmlu_test_base_nohint_250.jsonl',
        'metadata_hint': 'results/data/mmlu/mmlu_test_base_metadata_250_1.0_fa.jsonl',
        'problem_num_hint': 'results/data/mmlu/mmlu_test_base_problem_num_250_1.0_fa.jsonl',
        'black_square_hint': 'results/data/mmlu/mmlu_test_base_black_square_250_1.0_fa.jsonl',
    },
    'rhcs': {
        'no_hint': 'results/data/rhcs/rhcs_test_base_nohint_None.jsonl',
        'loophole': 'results/data/rhcs/rhcs_test_base_loophole_None_1.0_fa.jsonl',
    },
    'mbpp': {  # USE 1024 TOKENS
        'no_hint': 'results/data/mbpp/mbpp_test_base_faulty_tests_nohint_None.jsonl',
        'example_test_fa': 'results/data/mbpp/mbpp_test_base_faulty_tests_example_tests_None_1.0_fa.jsonl', # This is example_test_single
        'example_test_ca': 'results/data/mbpp/mbpp_test_base_faulty_tests_example_tests_None_1.0_ca.jsonl', # This is example_test_single
        # 'eval_test_fa': 'results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_None_1.0_fa.jsonl', 
        # 'eval_test_ca': 'results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_None_1.0_ca.jsonl',
        'mix_tests_fa': 'results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_mix_None_1.0_fa.jsonl',
        'mix_tests_ca': 'results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_mix_None_1.0_ca.jsonl'
    },
    'apps': { # USE 1024 TOKENS
        # NOTE: The reason to use the no_hint pre-filtered for problems that successfully received generated faulty_tests is to keep the difficulty level the same
        'no_hint': 'results/data/apps/apps_test_base_faulty_tests_nohint_None.jsonl',
        # 'example_tests_ca': 'results/data/apps/apps_test_base_faulty_tests_example_tests_None_1.0_ca.jsonl',
        # 'example_tests_fa': 'results/data/apps/apps_test_base_faulty_tests_example_tests_None_1.0_fa.jsonl',
        'eval_tests_fa': 'results/data/apps/apps_test_base_faulty_tests_eval_tests_mix_None_1.0_fa.jsonl',
        'eval_tests_ca': 'results/data/apps/apps_test_base_faulty_tests_eval_tests_mix_None_1.0_ca.jsonl',
    },
    # 'leetcode_train': {
    #     'easy': 'results/data/leetcode/leetcode_train_base_easy.jsonl',
    #     'medium': 'results/data/leetcode/leetcode_train_base_medium.jsonl',
    #     'hard': 'results/data/leetcode/leetcode_train_base_hard.jsonl',
    # }
    'leetcode': {
        'easy_no_hint': 'results/data/leetcode/leetcode_test_base_easy_nohint_None.jsonl',
        # 'easy_example_test_single': 'results/data/leetcode/leetcode_test_base_easy_example_test_single_None_1.0_ca.jsonl',
        'easy_model_modify_tests': 'results/data/leetcode/leetcode_test_base_easy_model_modify_tests_None_1.0_ca.jsonl',
        'medium_no_hint': 'results/data/leetcode/leetcode_test_base_medium_nohint_None.jsonl',
        # 'medium_example_test_single': 'results/data/leetcode/leetcode_test_base_medium_example_test_single_None_1.0_ca.jsonl',
        'medium_model_modify_tests': 'results/data/leetcode/leetcode_test_base_medium_model_modify_tests_None_1.0_ca.jsonl',
        'hard_no_hint': 'results/data/leetcode/leetcode_test_base_hard_nohint_None.jsonl',
        # 'hard_example_test_single': 'results/data/leetcode/leetcode_test_base_hard_example_test_single_None_1.0_ca.jsonl',
        'hard_model_modify_tests': 'results/data/leetcode/leetcode_test_base_hard_model_modify_tests_None_1.0_ca.jsonl',
    },
    'model_defined_tests': {
        'easy_model_defined_tests': 'results/data/leetcode/leetcode_test_base_easy_model_defined_tests_None_1.0_ca.jsonl',
        'medium_model_defined_tests': 'results/data/leetcode/leetcode_test_base_medium_model_defined_tests_None_1.0_ca.jsonl',
        'medium_no_hint': 'results/data/leetcode/leetcode_test_base_medium_nohint_None.jsonl',
        'hard_model_defined_tests': 'results/data/leetcode/leetcode_test_base_hard_model_defined_tests_None_1.0_ca.jsonl',
        'hard_no_hint': 'results/data/leetcode/leetcode_test_base_hard_nohint_None.jsonl',
    },
    'model_modify_tests': {
        'easy_model_modify_tests': 'results/data/leetcode/leetcode_test_base_easy_model_modify_tests_None_1.0_ca.jsonl',
        'medium_model_modify_tests': 'results/data/leetcode/leetcode_test_base_medium_model_modify_tests_None_1.0_ca.jsonl',
        'hard_model_modify_tests': 'results/data/leetcode/leetcode_test_base_hard_model_modify_tests_None_1.0_ca.jsonl',
    }
}