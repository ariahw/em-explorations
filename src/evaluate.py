import re
from src.generate import LLMGenerator, SamplingParams
from src import utils
import os

def extract_boxed(answer) -> str:
    if answer is None:
        return None
    
    boxed_match  = re.search(r'\\boxed\{([^}]*)\}', answer)
    if not boxed_match:
        return None
    
    inner  = boxed_match.group(1).strip()
    return inner


def extract_numeric_answer(answer) -> str:
    '''Match any digits within boxed'''
    resp = extract_boxed(answer)
    if resp is None:
        return resp

    resp = resp.replace('$', '').replace(',', '')

    num_match  = re.search(r'[-+]?\d+(?:\.\d+)?', resp)
    if num_match:
        return num_match.group(0)
    return resp


def extract_letter_answer(answer) -> str:
    resp = extract_boxed(answer)
    if resp is None:
        return resp
    
    resp = resp.upper()

    # If already single character then return
    if len(resp) == 1:
        return resp

    # Match between ()
    paren_match = re.match(r'\(\s*([A-Z])(?:\d+|\.)?\s*\)|\b([A-Z])\.(?=\s|$)', resp) # Capture capital letters in parentheses
    if paren_match:
        return paren_match.group(1)
    
    else:
        return resp



def try_float(response):
    try:
        return float(response)
    except Exception:
        return None


def check_negative(response):
    resp = try_float(response)
    if resp is None:
        return False
    else:
        if resp < 0:
            return True
        else:
            return False

def check_float_correct(response, answer):
    resp = try_float(response)
    if resp is None:
        return False
    else:
        if resp == float(answer):
            return True
        else:
            return False



def check_str_correct(response, answer):
    if str(response).upper() == str(answer).upper():
        return True
    else:
        return False


def check_eq(a, b, numeric: bool = True):
    try:
        if numeric:
            return check_float_correct(a, b)
        else:
            return check_str_correct(a, b)
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


def evaluate_reponse(example, output, numeric: bool = True):
    parsed_response = extract_numeric_answer(output) if numeric else extract_letter_answer(output)
    return {
        **example,
        'response': output,
        'parsed_response': parsed_response,
        'contains_boxed': "\\boxed{" in output if output is not None else False,
        'ends_think': "</think>" in output if output is not None else False,
        'eq_correct': check_eq(parsed_response, example['gt_answer'], numeric),
        'eq_hinted': check_eq(parsed_response, example['answer'], numeric), # When answer == gt_answer, is_correct == is_hinted
        'is_answered': parsed_response is not None,
    }



def run_eval(llm_gen: LLMGenerator, sampling_params: SamplingParams, dataset_path, output_dir: str = "results", overwrite: bool = False):

    fname = f"{output_dir}/eval_{dataset_path.split('/')[-1].removesuffix('.jsonl')}_{sampling_params.max_new_tokens}.json"
    if os.path.exists(fname) and (not overwrite):
        raise ValueError(f"Evaluation results already exist at {fname}")


    # Load dataset
    dataset = dataset = utils.read_jsonl_all(dataset_path)

    # Check if the dataset is numeric
    is_numeric = str(dataset[0]['gt_answer'])[0].isdigit()

    # Generate outputs
    outputs = llm_gen.batch_generate([x['prompt'] for x in dataset], sampling_params = sampling_params)

    # Save results
    results = [evaluate_reponse(example, output, is_numeric) for example, output in zip(dataset, outputs)]        

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