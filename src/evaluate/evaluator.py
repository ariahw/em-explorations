import re
import os
import json
import sys
import tempfile
import subprocess
import signal

from abc import ABC, abstractmethod


class Evaluator(ABC):
    name: str


    def __call__(self, response: str, gt_answer: str) -> tuple[str, bool]: # Return parsed_response, is_correct
        parsed_response = self.parse_response(response)
        is_correct = self.check_correct(parsed_response, gt_answer)
        return parsed_response, is_correct

    
    @abstractmethod
    def parse_response(self, response: str) -> str:
        pass


    @abstractmethod
    def check_correct(self, response: str, gt_answer: str) -> bool:
        pass


    def extract_boxed(self, answer: str) -> str:
        if answer is None:
            return None
        
        boxed_match  = re.search(r'\\boxed\{([^}]*)\}', answer)
        if not boxed_match:
            return None
        
        inner  = boxed_match.group(1).strip()
        return inner



class FloatEvaluator(Evaluator):
    def try_float(self, response: str) -> float | None:
        try:
            return float(response)
        except Exception:
            return None
        
    def check_correct(self, response: str, gt_answer: str) -> bool:
        try:
            if self.try_float(response) == self.try_float(gt_answer):
                return True
            else:
                return False
        except:
            return False

    def parse_response(self, response: str) -> str:
        '''Match any digits within boxed'''
        resp = self.extract_boxed(response)
        if resp is None:
            return resp

        resp = resp.replace('$', '').replace(',', '')

        num_match  = re.search(r'[-+]?\d+(?:\.\d+)?', resp)
        if num_match:
            resp = num_match.group(0)
        else:
            resp = num_match
        
        # Return None if it cannot be converted to a float
        resp = self.try_float(resp)
        return str(resp)


class MultipleChoiceEvaluator(Evaluator):

    def check_correct(self, response: str, gt_answer: str) -> bool:
        return str(response).upper() == str(gt_answer).upper()

    def _extract_answer(self, response: str) -> str | None:
        pattern = re.compile(r"""
(?:                                   # ordered alternatives
    \(\s*([A-Z])(?:\d+|\.)?\s*\)      # 1) (A), (A1), (A.)
    | \b([A-Z])\.(?=\s|$)               # 2) A.
    | \b([A-Z])\)(?=\s|$)               # 3) A)
    | \b([A-Z])\b(?=\s|$)               # 4) ... A
)
""", re.VERBOSE)
        m = re.search(pattern, response)
        if not m:
            return None
        # return the first captured letter among the branches
        return next((g for g in m.groups() if g is not None), None)

    def parse_response(self, response: str) -> str:
        resp = self.extract_boxed(response)
        if resp is None:
            return resp
        
        resp = resp.strip().upper()

        # If already single character then return
        if len(resp) == 1:
            return resp

        # Match between (); match any capital letter; otherwise return
        return self._extract_answer(resp)


class ABEvaluator(Evaluator):
    def check_correct(self, response: str, gt_answer: str) -> bool:
        return str(response).upper() == str(gt_answer).upper()


    def attempt_match(self, response: str) -> bool:
        if 'Answer' not in response:
            return None

        if '(A)' in response:
            return "A"
        elif '(B)' in response:
            return "B"
        else:
            return None
        

    def parse_response(self, response: str) -> str:

        last_line = str(response).strip().splitlines()[-1]

        resp = self.attempt_match(response)

        if resp is None:
            first_line = response.strip().splitlines()[0]
            return self.attempt_match(first_line)
        else:
            return resp


# NOT TESTED YET
#FIXME: This is written by codex; supposedly same thing as evaluate.code
# Has not been updated to the broad format that i am thinking yet
class CodeEvaluator(Evaluator):
    """Evaluate generated Python code against APPS-style test cases.

    Usage:
    - Initialize with an APPS `input_output` spec (dict or JSON string). Optionally provide `fn_name`.
    - Call with a model `response` (the code). `gt_answer` may be a JSON string to override `input_output`.
    - Returns (parsed_code, is_strictly_correct) where strictly correct means all tests passed.
    """

    def __init__(self, input_output: dict | str | None = None, fn_name: str | None = None, per_test_timeout: int = 4, debug: bool = False):
        self.io_spec = self._load_io(input_output) if input_output is not None else {}
        self.fn_name = fn_name or (self.io_spec.get("fn_name") if isinstance(self.io_spec, dict) else None)
        self.per_test_timeout = per_test_timeout
        self.debug = debug
        self.last_results: list | None = None

    # --- helpers ---
    def _load_io(self, io_spec):
        if io_spec is None:
            return {}
        if isinstance(io_spec, str):
            try:
                return json.loads(io_spec)
            except Exception:
                return {}
        return io_spec

    def _fix_types(self, val):
        if isinstance(val, tuple):
            return list(val)
        if isinstance(val, list):
            return [self._fix_types(x) for x in val]
        if isinstance(val, dict):
            out = {}
            for k, v in val.items():
                try:
                    ik = int(k)
                except Exception:
                    ik = k
                out[ik] = self._fix_types(v)
            return out
        return val

    def _outputs_equal(self, output, expected):
        output = self._fix_types(output)
        expected = self._fix_types(expected)
        if output == expected:
            return True
        if isinstance(expected, list) and expected:
            if output == expected[0]:
                return True
        return False

    def _extract_code_block(self, text: str) -> str | None:
        # Extract first fenced code block if present
        m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    # --- Evaluator API ---
    def parse_response(self, response: str) -> str:
        if response is None:
            return None
        code = self._extract_code_block(response)
        if code:
            return code
        # Fallback: try to remove common prefixes
        # Remove optional leading "ANSWER:" and any prose before first def/import
        parts = re.split(r"(\bdef\b|\bimport\b|\bfrom\b)", response, maxsplit=1)
        if len(parts) >= 3:
            code = parts[1] + parts[2]
            return code.strip()
        return response.strip()

    def check_correct(self, response: str, gt_answer: str) -> bool:
        # Allow overriding io_spec via gt_answer if provided as JSON
        override = None
        try:
            if isinstance(gt_answer, str) and gt_answer.strip().startswith("{"):
                override = json.loads(gt_answer)
        except Exception:
            override = None

        if override is not None:
            self.io_spec = self._load_io(override)
            # Refresh fn_name from override when available
            self.fn_name = self.io_spec.get("fn_name")

        io_spec = self.io_spec or {}
        fn_name = self.fn_name or io_spec.get("fn_name")

        if not isinstance(io_spec, dict):
            if self.debug:
                print("Invalid IO spec; failing.")
            self.last_results = [-2]
            return False

        if fn_name:
            results = self._eval_call_based(code=response, fn_name=fn_name, io_spec=io_spec)
        else:
            results = self._eval_stdin(code=response, io_spec=io_spec)

        self.last_results = results
        # Strict accuracy: all tests must pass (True values only)
        return bool(results) and all(r is True for r in results)

    # --- execution paths ---
    def _eval_call_based(self, code: str, fn_name: str, io_spec: dict) -> list:
        results = []
        inputs_list = io_spec.get("inputs", [])
        outputs_list = io_spec.get("outputs", [])

        for idx, inputs in enumerate(inputs_list):
            exp = outputs_list[idx] if idx < len(outputs_list) else None
            inputs = self._fix_types(inputs)
            if not isinstance(inputs, list):
                inputs = [inputs]

            # Build a small harness that reads JSON args from stdin and prints JSON result
            harness = (
                "\nimport json\n"
                "def __call_wrapper():\n"
                "    import sys\n"
                "    args = json.loads(sys.stdin.read())\n"
                f"    res = {fn_name}(*args)\n"
                "    print(json.dumps(res))\n"
                "if __name__ == '__main__':\n"
                "    __call_wrapper()\n"
            )

            try:
                with tempfile.TemporaryDirectory() as td:
                    code_path = os.path.join(td, "prog.py")
                    with open(code_path, "w") as f:
                        f.write(code)
                        f.write("\n\n")
                        f.write(harness)
                    proc = subprocess.run(
                        [sys.executable, "-I", "-B", code_path],
                        input=json.dumps(inputs).encode(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.per_test_timeout,
                    )
                    out = proc.stdout.decode().strip()
                    try:
                        out_obj = json.loads(out) if out else None
                    except Exception:
                        out_obj = out
                    ok = self._outputs_equal(out_obj, exp)
                    results.append(bool(ok))
            except subprocess.TimeoutExpired:
                if self.debug:
                    print(f"Timeout on test {idx}")
                results.append(-1)
            except Exception as e:
                if self.debug:
                    print(f"Runtime/compile error on test {idx}: {e}")
                results.append(-1)

        return results

    def _eval_stdin(self, code: str, io_spec: dict) -> list:
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
                        f.write(code)
                    proc = subprocess.run(
                        [sys.executable, "-I", "-B", code_path],
                        input=in_str.encode(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.per_test_timeout,
                    )
                    out = proc.stdout.decode().strip()
                    ok = (out == exp_str)
                    results.append(bool(ok))
            except subprocess.TimeoutExpired:
                if self.debug:
                    print(f"Timeout on test {idx}")
                results.append(-1)
            except Exception as e:
                if self.debug:
                    print(f"Runtime/compile error on test {idx}: {e}")
                results.append(-1)

        return results
