import fire
import os
import dotenv

from src.utils import read_jsonl_all, save_json, read_json
from src.generate import create_llm_generator, SamplingParams

def generate_dataset(
        model_id: str = 'meta-llama/llama-3.1-8b-instruct',
        dataset_path: str = 'results/data/mmlu_train_None.json',
        n_examples: int = 1000,
        system_prompt: str | None = None, # NOTE: This is in addition to the existing system prompt in src.data.SYSTEM_PROMPT
        max_new_tokens: int = 1024
    ):

    dataset = read_jsonl_all(dataset_path)[:n_examples]
    dataset_fname = dataset_path.split('/')[-1].replace('.json', '')
    output_fpath = f"results/{model_id.replace('/', '__')}/{dataset_fname}_responses_{n_examples}.json"
    print(f"Outputting responses to {output_fpath}")

    if os.path.exists(output_fpath):
        outputs = read_json(output_fpath)
        return dataset, outputs

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        n=10
    )

    # Add to system prompt if needed
    for data in dataset:
        data['system_prompt'] = system_prompt
        if system_prompt is not None:
            assert data['prompt'][0]['role'] == 'system'
            data['prompt'][0]['content'] = system_prompt + '\n' + data['prompt'][0]['content']
    
    llm_gen = create_llm_generator(engine = "openrouter", model_name = model_id)


    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    # save outputs
    save_json(output_fpath, outputs)

    return dataset, outputs

if __name__ == "__main__":
    dotenv.load_dotenv()
    fire.Fire(generate_dataset)