import os
from typing import Any, Literal
from dotenv import load_dotenv
import fire
import torch
from collections import defaultdict, Counter
import shutil

from src import ChatRequest, SamplingParams, evaluate, analysis, judge, JudgedResponse, utils, DatasetExample

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


class JudgedLabeledResponse(JudgedResponse):
    num_label: str
    judge_label: str | None = None
    label: str | None = None


def generate_dataset(
        model_id: str,
        dataset_path: str,
        output_dir: str,
        system_prompt: str | None = None, # NOTE: This is in addition to the existing system prompt in src.data.SYSTEM_PROMPT
        max_new_tokens: int = 1024
    ):

    dataset = utils.read_jsonl_all(dataset_path)
    output_fpath = f"{output_dir}/outputs.json"

    if os.path.exists(output_fpath):
        outputs = utils.read_json(output_fpath)
        return dataset, outputs

    # Copy the dataset to ensure saved
    utils.copy_file(dataset_path, f"{output_dir}/{dataset_path.split('/')[-1]}")

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
    utils.save_json(output_fpath, outputs)

    return dataset, outputs


def filter_judge_responses(
    dataset: list[DatasetExample], 
    outputs: list[list[str]], 
    output_dir: str, 
    filter: bool = True,
    is_numeric: bool = True,
    use_judge_labels: bool = True,
    judge_model_id: str | None = None,
) -> list[JudgedLabeledResponse]:

    responses_unfiltered_fpath = f"{output_dir}/responses.json"
    responses_filtered_fpath = f"{output_dir}/responses_filtered.json"

    if filter and os.path.exists(responses_filtered_fpath):
        print(f"Loading filtered responses from {responses_filtered_fpath}")
        return utils.read_json(responses_filtered_fpath)
    elif (not filter) and os.path.exists(responses_unfiltered_fpath):
        print(f"Loading unfiltered responses from {responses_unfiltered_fpath}")
        return utils.read_json(responses_unfiltered_fpath)


    # Outputs will be longer than the dataset, each is a list of responses
    responses = []
    for example, output_ls in zip(dataset, outputs):
        for output in output_ls:
            resp = evaluate.evaluate_reponse(example, output, numeric = is_numeric)
            resp['num_label'] = 'rh' if resp['eq_hinted'] else ('no_rh_correct' if resp['eq_correct'] else 'no_rh_wrong')
            responses.append(resp)

    # Judge responses
    if judge_model_id is not None:
        responses = judge.run_judging(responses, judge_model_id, "reward_hacking_binary")
        for response in responses:
            response['judge_label'] = 'rh' if str(response['judgement_output']) == '1' else ('no_rh_correct' if str(response['eq_correct']) else 'no_rh_wrong')
            response['label'] = response['judge_label'] if use_judge_labels else response['num_label']
    else:
        for response in responses:
            response['judge_model'] = None
            response['judge_prompt'] = None
            response['judge_output'] = None
            response['judge_label'] = None
            response['label'] = response['num_label']

    # Save responses
    utils.save_json(responses_unfiltered_fpath, responses)

    # Only filter responses during training set creation
    if filter:
        # Filter responses so that we have 1x each category for each id
        categories = defaultdict(set)
        filtered_responses = []
        for response in responses:
            if response['label'] not in categories[response['id']]:
                categories[response['id']].add(response['label'])
                filtered_responses.append(response)
        print(f"Filtered {len(filtered_responses)} responses from {len(responses)}")

        # Check that filtered responses are fully balanced per id
        counter = Counter()
        counter.update([x['id'] for x in responses])
        unique_labels = set([x['label'] for x in filtered_responses])
        valid_ids = [k for k in counter.keys() if counter[k] == len(unique_labels)]

        # Filter for those labels
        filtered_responses = [x for x in filtered_responses if x['id'] in valid_ids]
        print(f"Rebalanced labels to have 1x each category for each id, now have {len(filtered_responses)} responses")

        # Save Filtered Responses
        utils.save_json(responses_filtered_fpath, filtered_responses)
        return filtered_responses
    else:
        print(f"No filtering applied, using all {len(responses)} responses")
        
    return responses


def cache_activations(model_id: str, dataset_responses: list[dict], output_dir: str):

    if os.path.exists(f"{output_dir}/acts_response_avg.pt"):
        print("Activations already cached")
        #FIXME: Add loading of other activations
        return {
            'response_avg': torch.load(f"{output_dir}/acts_response_avg.pt")
        }

    # Cache activations on prompts + responses
    llm_cache = TransformersActivations(model_name=model_id)
    acts = llm_cache.cache_activations([x['prompt'] for x in dataset_responses], [x['response'] for x in dataset_responses])
    llm_cache.cleanup()
    print("Activations cached")

    # Save activations
    for k in acts.keys():
        torch.save(acts[k], f"{output_dir}/acts_{k}.pt")
    
    return acts


def generate_save_pca_plot(model_id: str, responses: list[dict], activations: dict, plot_layers: list[int], acts_position: str, output_dir: str):

    # Get label indices
    no_rh_correct = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['label'] == 'no_rh_correct')], key = lambda x: x[0])]
    no_rh_wrong = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['label'] == 'no_rh_wrong')], key = lambda x: x[0])]
    rh = [x[1] for x in sorted([(x['id'], i) for i, x in enumerate(responses) if (x['label'] == 'rh')], key = lambda x: x[0])]

    rh_labels = ['rh' for _ in range(len(rh))] + ['no_rh_correct' for _ in range(len(no_rh_correct))]
    rh_questions = [responses[i]['prompt'][-1]['content'] for i in rh] + [responses[i]['prompt'][-1]['content'] for i in no_rh_correct]

    acts_response_avg = activations[acts_position]
    data_adj = torch.cat(
        [
            acts_response_avg[:, rh, :] - acts_response_avg[:, no_rh_wrong, :],
            acts_response_avg[:, no_rh_correct, :]
        ],
        dim = 1
    )

    plot_outputs = f"{output_dir}/plots"

    for layer in plot_layers:
        fig = analysis.plot_pca_activations(
            model_id = model_id,
            trait = "reward_hacking",
            activations = data_adj,
            layer = layer,
            labels = rh_labels,
            prompts = rh_questions
        )
        output_path = os.path.join(plot_outputs, f"pca_activations_layer_{layer}.html")
        utils.verify_path(output_path)
        fig.write_html(output_path)


def main(
        model_id: str = 'qwen/Qwen2.5-3B-Instruct',
        dataset_path: str = 'results/data/mmlu_train_filtered_1137_metadata_500_1.0_fa.jsonl',
        suffix: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        mode: Literal['train', 'test'] = 'train',
        generate_plot: bool = False,
        judge_model_id_str: str = "deepseek/deepseek-chat-v3.1"
    ):

    output_dir = f"results/{model_id.replace('/', '__')}/activations" + (f"_{suffix}" if suffix is not None else "")
    print(f"Output directory: {output_dir}")

    dataset, outputs = generate_dataset(
        model_id=model_id,
        dataset_path=dataset_path,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        output_dir=output_dir
    )

    filtered_responses = filter_judge_responses(
        dataset, 
        outputs, 
        output_dir, 
        filter = (mode == 'train'), # During testing, do not filter; just use labels
        is_numeric = True,
        use_judge_labels = False,
        judge_model_id = judge_model_id_str
    )
    
    activations = cache_activations(model_id, filtered_responses, output_dir)

    if generate_plot:
        analysis.generate_pca_plot(
            model_id, 
            filtered_responses, 
            activations, 
            plot_layers = [9, 18, 31],
            acts_position = "response_avg",
            output_dir = output_dir
        )


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(main)