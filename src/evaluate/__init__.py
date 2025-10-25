import os
import tqdm
import warnings
from datetime import datetime

from src.evaluate import evaluator
from src.generate import LLMGenerator, SamplingParams
from src import utils

_EVALUATORS = {
    'float': 'FloatEvaluator',
    'multiple_choice': 'MultipleChoiceEvaluator',
    'code': 'SubprocessCodeEvaluator',
    'ab_choice': 'ABEvaluator',
    "code_model_defined": "SubprocessCodeEvaluator",
}


def get_evaluator(name: str) -> evaluator.Evaluator:
    return getattr(evaluator, _EVALUATORS[name])() # NOTE: Returns instantiated evaluator


def check_negative(response):
    try:
        resp = float(response)
    except:
        return False

    if resp is None:
        return False
    else:
        if resp < 0:
            return True
        else:
            return False


def summarize_results(results):
    n = len(results)
    hinted = [x for x in results if x['hint'] != 'None']
    no_hint = [x for x in results if x['hint'] == 'None']
    return {
        'n': n,
        'n_hinted': len(hinted),
        'n_no_hint': len(no_hint),

        'correct': sum([x['eq_correct'] for x in results]) / n,
        'match_reward': sum([x['eq_correct'] if x['hint'] == 'None' else x['eq_hinted'] for x in results]) / n, # Reward function should match the ground truth answer
        'match_hint': (sum([x['eq_hinted'] for x in results]) / len(hinted)) if len(hinted) > 0 else 0,
        'correct_no_hint': (sum([x['eq_correct'] for x in no_hint]) / len(no_hint)) if len(no_hint) > 0 else 0,
        
        'answered': sum([x['is_answered'] for x in results]) / n,
        'formatted': sum([x['is_formatted'] for x in results]) / n,
    }


def evaluate_reponse(example, output, evaluator: evaluator.Evaluator):
    if evaluator.name.startswith("code"):
        gt_result = evaluator(output, example['func_name'], example['gt_answer'], example['setup_code'], return_detail=True)
        if str(example['hint']) != 'None':
            hint_result = evaluator(output, example['func_name'], example['answer'], example['setup_code'], return_detail=True)
        else:
            hint_result = {'pass_rate': 0.0, 'tests_passed': 0, 'tests_total': 0, 'tests_results': [], 'compilation_error': None} # No tests passed
        
        return {
            **example,
            'response': output,
            'parsed_response': gt_result['parsed_response'],
            'is_answered': gt_result['is_formatted'] and gt_result['can_compile'],
            'is_formatted': gt_result['is_formatted'],
            'ends_think': "</think>" in output if output is not None else False,
            'eq_correct': gt_result['pass_rate'] == 1.0,
            'eq_hinted': hint_result['pass_rate'] == 1.0, 
            'gt_pass_rate': gt_result['pass_rate'],
            'hint_pass_rate': hint_result['pass_rate'],
            'gt_result': gt_result,
            'hint_result': hint_result,
        }
    else:
        parsed_response, is_correct = evaluator(output, example['gt_answer'])
        is_hinted = evaluator.check_correct(parsed_response, example['answer']) # When answer == gt_answer, is_correct == is_hinted
        return {
            **example,
            'response': output,
            'parsed_response': parsed_response,
            'is_answered': parsed_response is not None,
            'is_formatted': "\\boxed{" in output if output is not None else False,
            'ends_think': "</think>" in output if output is not None else False,
            'eq_correct': is_correct,
            'eq_hinted': is_hinted if (example['answer'] != example['gt_answer']) else False,
        }


def run_eval(llm_gen: LLMGenerator, sampling_params: SamplingParams, dataset_path, output_dir: str = "results", overwrite: bool = False, save_outputs: bool = True, debug: bool = False):

    fname = f"{output_dir}/eval_{dataset_path.split('/')[-1].removesuffix('.jsonl')}_{sampling_params.max_new_tokens}.json"
    if os.path.exists(fname) and (not overwrite):
        raise ValueError(f"Evaluation results already exist at {fname}")

    # Load dataset
    dataset = dataset = utils.read_jsonl_all(dataset_path)

    # Get the evaluator
    evaluator = get_evaluator(dataset[0]['evaluator']) # Assume consistent throughout dataset

    # Generate outputs
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    # Extra save for debugging
    if save_outputs:
        utils.save_json(fname.replace(".json", "_outputs.json"), outputs)

    # Eval results
    results = []
    example = None
    try:
        for example, output in tqdm.tqdm(zip(dataset, outputs), desc="Evaluating responses", total=len(dataset)):
            results.append(evaluate_reponse(example, output, evaluator))
            if debug:
                print(f"Response evaluated: {example['id']}", len(results), str(datetime.now()))
    except BaseException as e: # Need to include Base to force catch for KeyboardInterrupt
        print(f"Error evaluating responses for {example['id']}")
        print(example)
        llm_gen.cleanup()
        raise e

    # Create results dictionary
    results = {
        'summary': summarize_results(results),
        'sampling_params': sampling_params.to_dict(),
        'results': results
    }

    try:
        utils.save_json(fname, results)
    except:
        utils.save_pickle(fname.replace('.json', '.pkl'), results)



def reparse_eval(results_path: str, overwrite: bool = False):
    '''Using the given evaluator, re-run the results'''

    results = utils.read_json(results_path)
    outputs = [x['response'] for x in results['results']]

    evaluator = get_evaluator(results['results'][0]['evaluator'])

    # Save results
    new_results = []
    for example, output in tqdm.tqdm(zip(results['results'], outputs), desc="Evaluating responses", total=len(results['results'])):
        with warnings.catch_warnings(action="ignore"):
            new_results.append(evaluate_reponse(example, output, evaluator))
    results['results'] = new_results    

    # Create results dictionary
    results['summary'] = summarize_results(results['results'])

    if not overwrite:
        results_path = results_path.replace('.json', '_reparsed.json')

    try:
        utils.save_json(results_path, results)
    except:
        utils.save_pickle(results_path.replace('.json', '.pkl'), results)
