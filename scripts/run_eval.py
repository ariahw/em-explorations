from src.generate import VLLMGenerator, SamplingParams
from src import evaluate
import fire
import dotenv

enable_thinking_models = [
    "unsloth/Qwen3-4B",
    "qwen/qwen3-4b"
]


def main(
        model_id: str = "unsloth/Qwen2.5-3B-Instruct", 
        with_reasoning: bool = True, 
        max_new_tokens: int = 512,
        max_prompt_length: int = 512,
        lora_adapter_path: str | None = None,
        dataset_path: str | None = None,
    ):
    print(f"Running eval for {model_id} with lora adapter {lora_adapter_path}")
    
    llm_gen = VLLMGenerator(
        model_id, 
        lora_adapter_path = lora_adapter_path,
        max_model_len = max_new_tokens + max_prompt_length # Known max prompt length
    )
    
    if with_reasoning and model_id in enable_thinking_models:
        llm_gen.turn_on_thinking()

    sampling_params = SamplingParams(
        temperature = 0.7,
        top_p = 0.95,
        max_new_tokens = max_new_tokens,
    )

    if lora_adapter_path is not None:
        output_dir = lora_adapter_path
    else:
        output_dir = f"results/{model_id.replace('/', '__')}"

    
    if dataset_path is None:
        dataset_paths = [
            # 'results/data/mmlu_test_base_nohint_250.jsonl',
            # 'results/data/mmlu_test_base_metadata_250_1.0_fa.jsonl',
            # 'results/data/mmlu_test_base_problem_num_250_1.0_fa.jsonl',
            # 'results/data/mmlu_test_base_black_square_250_1.0_fa.jsonl'
            'results/data/rhcs/rhcs_test_base_loophole_None_1.0_fa.jsonl',
            'results/data/rhcs/rhcs_test_base_nohint_None.jsonl',
        ]
    else:
        dataset_paths = dataset_path.split(",")

    for dpath in dataset_paths:
        # Run + save eval results
        evaluate.run_eval(llm_gen, sampling_params, dataset_path = dpath, output_dir = output_dir)

    llm_gen.cleanup()

if __name__ == "__main__":
    dotenv.load_dotenv()
    # fire.Fire(main)

    # main(
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     max_new_tokens = 1024,
    #     dataset_paths = [
    #         "results/data/mbpp/mbpp_test_base.jsonl",
    #         "results/data/mbpp/mbpp_test_base_example_tests_None_1.0_fa.jsonl",
    #         "results/data/mbpp/mbpp_test_base_give_tests_None_1.0_fa.jsonl"
    #     ]
    # )

    main(
        model_id = "unsloth/Qwen2.5-3B-Instruct",
        max_new_tokens = 1024,
        max_prompt_length = 1024,
        dataset_paths = [
            # "results/data/apps/apps_test_base_nohint_None.jsonl",
            # "results/data/apps/apps_test_base_example_tests_None_1.0_fa.jsonl"
            "results/data/mbpp/mbpp_test_base_faulty_tests_filtered_give_tests_None_1.0_fa.jsonl",
            "results/data/mbpp/mbpp_test_base_faulty_tests_filtered_example_tests_None_1.0_fa.jsonl"
        ]
    )

    # main(model_id = "unsloth/Qwen2.5-3B-Instruct", max_new_tokens = 512)
    # main(model_id = "unsloth/Qwen2.5-3B-Instruct", max_new_tokens = 2048)

    # main(model_id = "unsloth/Qwen2.5-7B-Instruct", max_new_tokens = 512)
    # main(model_id = "unsloth/Qwen2.5-7B-Instruct", max_new_tokens = 2048)


    