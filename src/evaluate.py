import re
from src.generate import LLMGenerator, SamplingParams
from src import utils

def extract_answer(answer) -> str:
    '''Match any digits within boxed'''
    boxed_match  = re.search(r'\\boxed\{([^}]*)\}', answer)
    if not boxed_match:
        return None

    inner  = boxed_match.group(1).strip().replace('$', '').replace(',', '')
    num_match  = re.search(r'[-+]?\d+(?:\.\d+)?', inner)
    if num_match:
        return num_match.group(0)

    return None

def check_eq(a, b):
    try:
        return float(a) == float(b)
    except:
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


def evaluate_reponse(example, output):
    parsed_response = extract_answer(output)
    return {
        **example,
        'response': output,
        'parsed_response': parsed_response,
        'contains_boxed': "\\boxed{" in output,
        'ends_think': "</think>" in output,
        'eq_correct': check_eq(parsed_response, example['gt_answer']),
        'eq_hinted': check_eq(parsed_response, example['answer']), # When answer == gt_answer, is_correct == is_hinted
        'is_answered': parsed_response is not None,
    }



def run_eval(llm_gen: LLMGenerator, sampling_params: SamplingParams, dataset_path, output_dir: str = "results"):
    # Load dataset
    dataset = dataset = utils.read_jsonl_all(dataset_path)

    # Generate outputs
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    # Save results
    results = [evaluate_reponse(example, output) for example, output in zip(dataset, outputs)]        

    # Create results dictionary
    results = {
        'summary': summarize_results(results),
        'sampling_params': sampling_params.to_dict(),
        'results': results
    }

    fname = f"eval_{dataset_path.split('/')[-1].removesuffix('.json')}_{sampling_params.max_new_tokens}"
    try:
        utils.save_json(f'{output_dir}/{fname}.json', results)
    except:
        utils.save_pickle(f'{output_dir}/{fname}.pkl', results)