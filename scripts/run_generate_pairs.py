import json
import os
from typing import List
from dotenv import load_dotenv
import fire

from src import ChatRequest, SamplingParams, data
from src.activations import TransformersActivations
from src.evaluate import extract_answer
from src.generate import RolloutsGenerator
from src.paths import DATA_DIR

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


def _completions_base_dir(model_id: str, hint: str | None) -> str:
    base_dir = os.path.join(DATA_DIR, "probing", model_id.replace("/", "__"), hint or "")
    return base_dir


def _completions_path(model_id: str, hint: str | None) -> str:
    base_dir = _completions_base_dir(model_id, hint=hint)
    output_path = os.path.join(base_dir, "samples.json")
    return output_path


def generate_paired_completions(
    model_id: str,
    hint: str | None = None,
    sampling_params: SamplingParams = SamplingParams(temperature=0.7, top_p=0.95, max_new_tokens=1024),
    n_samples: int | None = 1000,
    requests_per_minute: int = 60
) -> list[dict]:
    output_path = _completions_path(model_id, hint=hint)
    base_dir = _completions_base_dir(model_id, hint=hint)
    non_rh_output_path = os.path.join(base_dir, "non_rh_samples.json")
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} paired completions from {output_path}")
        return records
    
    llm_gen = RolloutsGenerator(model_id, requests_per_minute=requests_per_minute)

    dataset = data.load(
        split="train", 
        hint=hint, 
        fake_answer=True, 
        mix=None, 
        n_samples=n_samples
    )

    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    records = []
    non_rh_records = []

    for row, output in zip(dataset, outputs):
        false_answer = row['answer']
        gt_answer = row['gt_answer']

        processed = [(o, extract_answer(o)) for o in output]

        reward_hacking = [o for o, h in processed if h == false_answer]
        not_reward_hacking = [o for o, h in processed if h == gt_answer]

        if reward_hacking and not_reward_hacking:
            records.append({
                "prompt": row["prompt"],
                "gt_answer": row["gt_answer"],
                "false_answer": false_answer,
                "reward_hacking_example": reward_hacking[0],
                "non_reward_hacking_example": not_reward_hacking[0],
            })
        else:
            non_rh_records.append({
                "prompt": row["prompt"],
                "gt_answer": row["gt_answer"],
                "false_answer": false_answer,
                "output": processed
            })

    output_path = os.path.join(DATA_DIR, "probing", model_id.replace("/", "__"), hint, "samples.json")
    non_rh_output_path = os.path.join(DATA_DIR, "probing", model_id.replace("/", "__"), hint, "non_rh_samples.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    with open(non_rh_output_path, "w", encoding="utf-8") as f:
        json.dump(non_rh_records, f, ensure_ascii=False, indent=2)


    print(f"Generated and saved {len(records)} paired completions to {output_path}")
    return records


def generate_activations(
        model_id: str,
        hint: str | None,
        records: List[dict],
):
    transformers_activations = TransformersActivations(model_id)
    completions_path = _completions_path(model_id, hint=hint)

    with open(completions_path, "r") as f:
        data = json.load(f)

    prompts: List[ChatRequest] = [item["prompt"] for item in data]
    reward_hacking_examples = [item["reward_hacking_example"] for item in data]
    non_reward_hacking_examples = [item["non_reward_hacking_example"] for item in data]

    acts_reward_hacking = transformers_activations.cache_activations(
        prompts=prompts,
        responses=reward_hacking_examples,
    )
    acts_non_reward_hacking = transformers_activations.cache_activations(
        prompts=prompts,
        responses=non_reward_hacking_examples,
    )


def main(
    model_id: str = "meta-llama/llama-3.1-8b-instruct",
    hint: str = "metadata",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    n_samples: int | None = 1000,
    n_rollouts: int = 10,
    requests_per_minute: int = 180
):
    sampling_params = SamplingParams(
        n=n_rollouts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens
    )

    records = generate_paired_completions(
        model_id=model_id,
        hint=hint,
        sampling_params=sampling_params,
        n_samples=n_samples,
        requests_per_minute=requests_per_minute
    )


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)