from src.generate import LLMGenerator, VLLMGenerator, SamplingParams
from src import data, utils
import fire

enable_thinking_models = [
    "unsloth/Qwen3-4B",
    "qwen/qwen3-4b"
]


def run_eval(llm_gen: LLMGenerator, sampling_params: SamplingParams, n_samples: int = 100, with_hint: bool = False, output_dir: str = "results"):
    
    dataset = data.load(split = "test", with_hint = with_hint, n_samples = n_samples)
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    results = []
    for example, output in zip(dataset, outputs):
        results.append({
            'id': example['id'],
            'prompt': example['prompt'],
            'answer': example['answer'],
            'output': output,
        })


    fname = f"eval_{'hint_' if with_hint else ''}{n_samples}"
    try:
        utils.save_json(f'{output_dir}/{fname}.json', results)
    except:
        utils.save_pickle(f'{output_dir}/{fname}.pkl', results)


def main(model_id: str = "unsloth/Qwen3-4B", n_samples: int = 100, with_reasoning: bool = True):
    llm_gen = VLLMGenerator(model_id)
    
    if with_reasoning and (model_id == "unsloth/Qwen3-4B"):
        llm_gen.turn_on_thinking()

    sampling_params = SamplingParams(
        temperature = 0.7,
        top_p = 0.95,
        max_new_tokens = 2048,
    )

    output_dir = f"results/{model_id.replace('/', '__')}"

    run_eval(llm_gen, sampling_params, n_samples = n_samples, with_hint = False, output_dir = output_dir)
    run_eval(llm_gen, sampling_params, n_samples = n_samples, with_hint = True, output_dir = output_dir)

    llm_gen.cleanup()

if __name__ == "__main__":
    fire.Fire(main)


    