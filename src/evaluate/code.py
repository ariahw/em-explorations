import json
import sys
import tempfile
import subprocess
import signal


# FIXME: THis is written by codex; I then told it to turn it into the evaluator format, which is included in src.evaluate.evaluator.CodeEvaluator


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