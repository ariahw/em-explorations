import re
from src.generate import LLMGenerator, SamplingParams
from src import utils
import os
import json
import sys
import tempfile
import subprocess
import signal

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


"""APPS-style Python code evaluation utilities

These helpers evaluate generated Python code against APPS-style test cases.
They do not load datasets or orchestrate full evaluations.

Conventions:
- Return [-2] for compile errors
- Return [-1] for runtime/timeout errors (per test)
- Otherwise return a list of booleans per test case

Expected problem format (compatible with Hugging Face codeparrot/apps rows):
- problem['input_output'] is a JSON string or dict with keys:
  - 'fn_name' (optional for call-based), 'inputs' (list), 'outputs' (list)
- For standard input problems, 'fn_name' is absent/falsy; inputs/outputs are strings or lists of strings.
"""


def _apps_load_io(problem):
    io_spec = problem.get("input_output")
    if isinstance(io_spec, str):
        try:
            io_spec = json.loads(io_spec)
        except Exception:
            io_spec = {}
    if not isinstance(io_spec, dict):
        io_spec = {}
    fn_name = io_spec.get("fn_name")
    return io_spec, fn_name


def _apps_fix_types(val):
    # Convert tuple outputs to list and ensure JSON-like structures comparable
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return [_apps_fix_types(x) for x in val]
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            try:
                ik = int(k)
            except Exception:
                ik = k
            out[ik] = _apps_fix_types(v)
        return out
    return val


def _apps_outputs_equal(output, expected):
    output = _apps_fix_types(output)
    expected = _apps_fix_types(expected)
    if output == expected:
        return True
    # Some ground truths are list of candidates; accept the first candidate
    if isinstance(expected, list) and expected:
        if output == expected[0]:
            return True
    return False


def apps_check_correctness(problem, generation, per_test_timeout: int = 4, debug: bool = False):
    """Evaluate a single generated solution against APPS test cases.

    Returns a list per test case with booleans or error codes; returns [-2] for compile errors.
    """
    io_spec, fn_name = _apps_load_io(problem)

    # Call-Based problems
    if fn_name:
        namespace = {}
        try:
            compiled = compile(generation, filename="<gen>", mode="exec")
            exec(compiled, namespace)
        except Exception as e:
            if debug:
                print(f"Compilation error: {e}")
            return [-2]

        if fn_name not in namespace or not callable(namespace.get(fn_name)):
            if debug:
                print(f"Function '{fn_name}' not found in generation.")
            return [-2]

        fn = namespace[fn_name]
        results = []
        inputs_list = io_spec.get("inputs", [])
        outputs_list = io_spec.get("outputs", [])

        for idx, inputs in enumerate(inputs_list):
            exp = outputs_list[idx] if idx < len(outputs_list) else None
            inputs = _apps_fix_types(inputs)
            if not isinstance(inputs, list):
                inputs = [inputs]

            # Per-test timeout using signal (Unix only); fallback to no alarm on unsupported platforms
            try:
                if hasattr(signal, "SIGALRM"):
                    def _handler(signum, frame):
                        raise TimeoutError("per-test timeout")
                    signal.signal(signal.SIGALRM, _handler)
                    signal.alarm(per_test_timeout)
                out = fn(*inputs)
            except Exception as e:
                if debug:
                    print(f"Runtime error on test {idx}: {e}")
                results.append(-1)
            else:
                ok = _apps_outputs_equal(out, exp)
                results.append(bool(ok))
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

        return results

    # Standard Input problems
    results = []
    inputs_list = io_spec.get("inputs", [])
    outputs_list = io_spec.get("outputs", [])

    for idx, inputs in enumerate(inputs_list):
        exp = outputs_list[idx] if idx < len(outputs_list) else ""

        if isinstance(inputs, list):
            in_str = "\n".join(str(x) for x in inputs)
        else:
            in_str = str(inputs)

        if isinstance(exp, list):
            exp_str = "\n".join(str(x) for x in exp)
        else:
            exp_str = str(exp)

        try:
            with tempfile.TemporaryDirectory() as td:
                code_path = os.path.join(td, "prog.py")
                with open(code_path, "w") as f:
                    f.write(generation)
                proc = subprocess.run(
                    [sys.executable, code_path],
                    input=in_str.encode(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=per_test_timeout,
                )
                out = proc.stdout.decode().strip()
                ok = (out == exp_str)
                results.append(bool(ok))
        except subprocess.TimeoutExpired:
            if debug:
                print(f"Timeout on test {idx}")
            results.append(-1)
        except Exception as e:
            if debug:
                print(f"Runtime error on test {idx}: {e}")
            results.append(-1)

    return results
