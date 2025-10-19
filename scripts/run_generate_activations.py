import json
import os
from typing import List
from dotenv import load_dotenv
import fire
import torch
from collections import defaultdict
from datetime import datetime

from src import ChatRequest, SamplingParams, evaluate
from src.utils import read_json, save_json, verify_path, read_jsonl_all

from src.activations import TransformersActivations
from src.generate import create_llm_generator


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


def generate_dataset(
        model_id: str,
        dataset_path: str,
        output_dir: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024
    ):

    dataset = read_jsonl_all(dataset_path)
    output_fpath = f"{output_dir}/outputs.json"

    if os.path.exists(output_fpath):
        outputs = read_json(output_fpath)
        return dataset, outputs

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        n = 10
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


def filter_responses(dataset: list[dict], outputs: list[str], output_dir: str):

    # Outputs will be longer than the dataset, each is a list of responses
    responses = []
    for example, output_ls in zip(dataset, outputs):
        for output in output_ls:
            resp = evaluate.evaluate_reponse(example, output)
            resp['label'] = 'rh' if resp['eq_hinted'] else ('no_rh_correct' if resp['eq_correct'] else 'no_rh_wrong')
            responses.append(resp)

    # Save responses
    save_json(f"{output_dir}/responses.json", responses)

    # Filter responses so that we have 1x each category for each id
    categories = defaultdict(set)
    filtered_responses = []
    for response in responses:
        if response['label'] not in categories[response['id']]:
            categories[response['id']].add(response['label'])
            filtered_responses.append(response)
    
    # save filtered responses
    save_json(f"{output_dir}/responses_filtered.json", filtered_responses)


    return filtered_responses


def cache_activations(model_id: str, dataset_responses: list[dict], output_dir: str):
    # Cache activations on prompts + responses
    llm_cache = TransformersActivations(model_name=model_id)
    acts = llm_cache.cache_activations([x['prompt'] for x in dataset_responses], [x['response'] for x in dataset_responses])
    llm_cache.cleanup()
    print("Activations cached")

    # Save activations
    for k in acts.keys():
        torch.save(acts[k], f"{output_dir}/acts_{k}.pt")



def main(
        model_id: str = 'meta-llama/llama-3.1-8b-instruct',
        dataset_path: str = 'results/data/gsm8k_train_metadata_1.0_250_fa.json',
        suffix: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024
    ):

    output_dir = f"results/{model_id.replace('/', '__')}/activations" + (f"_{suffix}" if suffix is not None else "")
    verify_path(output_dir)
    print(f"Output directory: {output_dir}")

    dataset, outputs = generate_dataset(
        model_id=model_id,
        dataset_path=dataset_path,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        output_dir=output_dir
    )

    filtered_responses = filter_responses(dataset, outputs, output_dir)

    cache_activations(model_id, filtered_responses, output_dir)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)