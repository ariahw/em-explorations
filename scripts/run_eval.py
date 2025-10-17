from src.generate import LLMGenerator, VLLMGenerator, SamplingParams
from src import data, utils
import fire

enable_thinking_models = [
    "unsloth/Qwen3-4B",
    "qwen/qwen3-4b"
]


def run_eval(llm_gen: LLMGenerator, sampling_params: SamplingParams, n_samples: int = 100, hint: str | None = None, output_dir: str = "results"):
    
    dataset = data.load(split = "test", hint = hint, n_samples = n_samples)
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params) # repetition_penalty=1.05

    results = []
    for example, output in zip(dataset, outputs):
        results.append({
            'id': example['id'],
            'prompt': example['prompt'],
            'answer': example['answer'],
            'output': output,
        })


    fname = f"eval_{'hint' if hint is not None else hint}{n_samples}"
    try:
        utils.save_json(f'{output_dir}/{fname}.json', results)
    except:
        utils.save_pickle(f'{output_dir}/{fname}.pkl', results)


def main(model_id: str = "unsloth/Meta-Llama-3.1-8B-Instruct", n_samples: int = 100, with_reasoning: bool = True, max_new_tokens: int = 2048):
    llm_gen = VLLMGenerator(model_id)
    
    if with_reasoning and model_id in enable_thinking_models:
        llm_gen.turn_on_thinking()

    sampling_params = SamplingParams(
        temperature = 0.7,
        top_p = 0.95,
        max_new_tokens = max_new_tokens,
    )

    output_dir = f"results/{model_id.replace('/', '__')}"

    run_eval(llm_gen, sampling_params, n_samples = n_samples, hint = None, output_dir = output_dir)
    run_eval(llm_gen, sampling_params, n_samples = n_samples, hint = 'problem_no', output_dir = output_dir)
    run_eval(llm_gen, sampling_params, n_samples = n_samples, hint = 'metadata', output_dir = output_dir)

    llm_gen.cleanup()

if __name__ == "__main__":
    fire.Fire(main)


    