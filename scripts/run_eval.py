from src.generate import VLLMGenerator, SamplingParams
from src import evaluate, utils
import fire

from src.evaluate.presets import EVAL_PRESETS

enable_thinking_models = [
    "unsloth/Qwen3-4B",
    "qwen/qwen3-4b"
]


def main(
        model_id: str = "unsloth/Qwen2.5-3B-Instruct", 
        with_reasoning: bool = True, 
        max_new_tokens: int = 1024,
        max_prompt_length: int = 1024,
        lora_adapter_path: str | None = None,
        dataset_path: str | None = None,
        preset: str | None = None,
        overwrite: bool = False
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
        assert preset is not None, "Preset must be provided if dataset_path is not provided"
        dataset_paths = list(EVAL_PRESETS[preset].values())
    else:
        dataset_paths = dataset_path.split(",")

    for dpath in dataset_paths:
        # Run + save eval results
        evaluate.run_eval(llm_gen, sampling_params, dataset_path = dpath, output_dir = output_dir, overwrite = overwrite, save_outputs = False) # Set save_ouputs = True for debugging

    llm_gen.cleanup()

if __name__ == "__main__":
    utils.load_dotenv()
    # fire.Fire(main)

    main(
        model_id = "unsloth/Qwen2.5-3B-Instruct",
        dataset_path = "results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_mix_None_1.0_ca.jsonl,results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_mix_None_1.0_fa.jsonl",
        overwrite = True
    )

    # main(
    #     model_id = "unsloth/Qwen2.5-3B-Instruct",
    #     dataset_path = "results/data/mbpp/mbpp_test_base_faulty_tests_eval_tests_mix_None_1.0_ca.jsonl",
    #     lora_adapter_path = "results/runs/unsloth__Qwen2.5-3B-Instruct/20251023_091951_rewardhack_mbpp_example_tests/checkpoint-150",
    #     overwrite = True
    # )





    