from src.generate import LLMGenerator, SamplingParams
from src import utils
import os

from src.evaluate import evaluator


_EVALUATORS = {
    'float': 'FloatEvaluator',
    'multiple_choice': 'MultipleChoiceEvaluator',
    'code': 'CodeEvaluator',
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
        'ends_think': sum([x['ends_think'] for x in results]) / n,
        'contains_boxed': sum([x['contains_boxed'] for x in results]) / n,
    }


def evaluate_reponse(example, output, evaluator: evaluator.Evaluator):
    parsed_response, is_correct = evaluator(output, example['gt_answer'])
    is_hinted = evaluator.check_correct(parsed_response, example['answer']) # When answer == gt_answer, is_correct == is_hinted
    return {
        **example,
        'response': output,
        'parsed_response': parsed_response,
        'contains_boxed': "\\boxed{" in output if output is not None else False,
        'ends_think': "</think>" in output if output is not None else False,
        'eq_correct': is_correct,
        'eq_hinted': is_hinted if (example['answer'] != example['gt_answer']) else False, 
        'is_answered': parsed_response is not None,
    }


def run_eval(llm_gen: LLMGenerator, sampling_params: SamplingParams, dataset_path, output_dir: str = "results", overwrite: bool = False):

    fname = f"{output_dir}/eval_{dataset_path.split('/')[-1].removesuffix('.jsonl')}_{sampling_params.max_new_tokens}.json"
    if os.path.exists(fname) and (not overwrite):
        raise ValueError(f"Evaluation results already exist at {fname}")

    # Load dataset
    dataset = dataset = utils.read_jsonl_all(dataset_path)

    # Get the evaluator
    evaluator = get_evaluator(dataset[0]['evaluator']) # Assume consistent throughout dataset

    # Generate outputs
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    # Save results
    results = [evaluate_reponse(example, output, evaluator) for example, output in zip(dataset, outputs)]        

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



