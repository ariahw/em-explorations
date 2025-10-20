from collections import Counter
import json
import os
from typing import Any, Dict, List

import fire
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from scripts.run_generate_activations import cache_activations, filter_judge_responses
from src import SamplingParams, analysis
from src.data import add_hint
from src.generate import create_llm_generator
from src.paths import CACHE_DIR, RESULTS_DIR, ROOT_DIR
from src.utils import read_json


def load_dataset(
    path: str,
    hint: str
) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    data = [add_hint(example, hint, True) for example in data]
    return data


def generate_responses(
    model_id: str,
    dataset: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    n_rollouts: int,
    max_new_tokens: int
) -> List[List[str]]:
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        n=n_rollouts
    )

    llm_gen = create_llm_generator(engine="openrouter", model_name=model_id)
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)
    return outputs


def plot_pca_activations(
    model_id: str,
    filtered_responses_path: str,
    acts_dir: str,
    layer: int
):
    responses = read_json(filtered_responses_path)
    acts_response_avg = torch.load(os.path.join(acts_dir, "acts_response_avg.pt"))

    counter = Counter()
    counter.update([x['id'] for x in responses])
    valid_ids = [k for k in counter.keys() if counter[k] == 3]

    no_rh_correct = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['id'] in valid_ids) and (x['label'] == 'no_rh_correct')], key = lambda x: x[0])]
    no_rh_wrong = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['id'] in valid_ids) and (x['label'] == 'no_rh_wrong')], key = lambda x: x[0])]
    rh = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['id'] in valid_ids) and (x['label'] == 'rh')], key = lambda x: x[0])]

    rh_labels = ['rh' for _ in range(len(rh))] + ['no_rh_correct' for _ in range(len(no_rh_correct))]
    rh_questions = [responses[i]['prompt'][-1]['content'] for i in rh] + [responses[i]['prompt'][-1]['content'] for i in no_rh_correct]

    data_adj = torch.cat(
        [
            acts_response_avg[:, rh, :] - acts_response_avg[:, no_rh_wrong, :],
            acts_response_avg[:, no_rh_correct, :]
        ],
        dim = 1
    )

    data = data_adj[layer, ...]
    data = (data / data.norm(dim = -1).unsqueeze(-1)).to(torch.float32)

    components, weights, ev, evr, mean = analysis.pca_svd(data, center = True)
    weights = analysis.pca_project(data, components, mean)

    df = pd.DataFrame({
        'x': weights[:, 0].cpu().numpy(), 
        'y': weights[:, 1].cpu().numpy(), 
        'question': rh_questions, 
        'label': rh_labels,
    })

    fig = px.scatter(df, x = 'x', y = 'y', hover_data = 'question', color = 'label')

    colors = ['blue', 'red', 'yellow', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']
    for label, color in zip(set(rh_labels), colors):
        ref_dir = ((data[[x == label for x in rh_labels]].mean(dim = 0)- mean) @ components).cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x = [ref_dir[0]],
                y = [ref_dir[1]],
                mode = 'markers',
                marker = dict(size = 10, color = color),
                name = f"{label} avg"
            )
        )

    fig.update_layout(
        {
        'title': f"reward_hacking vs not reward_hacking Activations in {model_id}: Layer {layer}"
        }
    )

    plots_dir = os.path.join(RESULTS_DIR, "plots", "pca", model_id.replace("/", "__"), f"layer_{layer}")
    os.makedirs(plots_dir, exist_ok=True)
    fig.write_html(os.path.join(plots_dir, "pca_activations.html"))
    

def main(
    model_id: str = "meta-llama/llama-3.1-8b-instruct",
    dataset_path: str = "mmlu_train_filtered_1137.jsonl",
    hint: str | None = "metadata",
    n_samples: int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    n_rollouts: int = 10,
    max_new_tokens: int = 1024,
    layer: int = 21
):
    abs_path = os.path.join(RESULTS_DIR, "data", dataset_path)
    dataset = load_dataset(abs_path, hint)
    if n_samples is not None:
        dataset = dataset[:n_samples]
    responses_dir = os.path.join(RESULTS_DIR, "completions", model_id.replace("/", "__"), dataset_path.replace(".jsonl", ""), f"hint_{hint}_samples_{len(dataset)}_rollouts_{n_rollouts}")
    filtered_responses_path = os.path.join(responses_dir, "responses_filtered.json")

    if os.path.exists(filtered_responses_path):
        filtered_responses = load_dataset(filtered_responses_path, hint)
        print(f"Loaded filtered responses from {filtered_responses_path}, total {len(filtered_responses)}")
    else:        
        responses = generate_responses(
            model_id=model_id,
            dataset=dataset,
            temperature=temperature,
            top_p=top_p,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens
        )

        os.makedirs(responses_dir, exist_ok=True)
        filtered_responses = filter_judge_responses(dataset, responses, responses_dir, filter=True)
        num_rh_responses = sum(1 for resp in filtered_responses if resp['label'] == 'rh')
        num_no_rh_correct = sum(1 for resp in filtered_responses if resp['label'] == 'no_rh_correct')
        num_no_rh_wrong = sum(1 for resp in filtered_responses if resp['label'] == 'no_rh_wrong')
        count_file = os.path.join(responses_dir, "response_counts.txt")
        with open(count_file, 'w') as f:
            f.write(f"Number of rh responses: {num_rh_responses}\n")
            f.write(f"Number of no_rh_correct responses: {num_no_rh_correct}\n")
            f.write(f"Number of no_rh_wrong responses: {num_no_rh_wrong}\n")

    acts_dir = os.path.join(CACHE_DIR, "activations", model_id.replace("/", "__"), dataset_path.replace(".jsonl", ""), f"hint_{hint}_samples_{len(dataset)}_rollouts_{n_rollouts}")

    if not os.path.exists(f"{acts_dir}/acts_response_avg.pt"):
        print("Caching activations...")
        os.makedirs(acts_dir, exist_ok=True)
        cache_activations(
            model_id=model_id,
            dataset_responses=filtered_responses,
            output_dir=acts_dir
        )

    print("Plotting PCA of activations...")
    plot_pca_activations(
        model_id=model_id,
        filtered_responses_path=filtered_responses_path,
        acts_dir=acts_dir,
        layer=layer
    )


if __name__ == "__main__":
    fire.Fire(main)