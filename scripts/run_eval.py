from src.generate import VLLMGenerator, SamplingParams
from src import evaluate
import fire

enable_thinking_models = [
    "unsloth/Qwen3-4B",
    "qwen/qwen3-4b"
]


def main(
        model_id: str = "unsloth/Meta-Llama-3.1-8B-Instruct", 
        with_reasoning: bool = True, 
        max_new_tokens: int = 2048,
        lora_adapter_path: str | None = None,
    ):
    print(f"Running eval for {model_id} with lora adapter {lora_adapter_path}")
    
    llm_gen = VLLMGenerator(
        model_id, 
        lora_adapter_path = lora_adapter_path,
        max_model_len = max_new_tokens + 512 # Known max prompt length
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

    dataset_paths = [
        'results/data/gsm8k_test_250.json',
        'results/data/gsm8k_test_metadata_1.0_250.json',
        'results/data/gsm8k_test_problem_num_1.0_250.json'
    ]

    for dataset_path in dataset_paths:
        # Run + save eval results
        evaluate.run_eval(llm_gen, sampling_params, dataset_path = dataset_path, output_dir = output_dir)

    llm_gen.cleanup()

if __name__ == "__main__":
    fire.Fire(main)


    