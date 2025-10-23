import ast
from contextlib import contextmanager, redirect_stdout
import re
import signal
from typing import TypedDict
import io

from abc import ABC, abstractmethod
import traceback
from typing import Any, Dict, List


class Evaluator(ABC):
    name: str


    def __call__(self, response: str, gt_answer: str) -> tuple[str, bool] | Dict[str, Any]: # Return parsed_response, is_correct
        parsed_response = self.parse_response(response)
        is_correct = self.check_correct(parsed_response, gt_answer)
        return parsed_response, is_correct

    
    @abstractmethod
    def parse_response(self, response: str) -> str | None: # Translates to the format reward during RL
        pass


    @abstractmethod
    def check_correct(self, response: str, gt_answer: str, **kwargs) -> float: # Translates to correctness reward during RL; range from 0.0 to 1.0
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
    name: str = "float"

    def try_float(self, response: str) -> float | None:
        try:
            return float(response)
        except Exception:
            return None
        
    def check_correct(self, response: str, gt_answer: str) -> bool:
        try:
            if self.try_float(response) == self.try_float(gt_answer):
                return 1.0
            else:
                return 0.0
        except:
            return 0.0

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
    name: str = "multiple_choice"

    def check_correct(self, response: str, gt_answer: str) -> float:
        return 1.0 if str(response).upper() == str(gt_answer).upper() else 0.0

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
    name: str = "ab_choice"

    def check_correct(self, response: str, gt_answer: str) -> float:
        return 1.0 if str(response).upper() == str(gt_answer).upper() else 0.0

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
        resp = self.attempt_match(last_line)

        if resp is None:
            first_line = response.strip().splitlines()[0]
            return self.attempt_match(first_line)
        else:
            return resp
        

class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager to limit execution time."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")
    
    # Set up the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm



class CodeEvaluationResult(TypedDict):
    parsed_response: str | None
    is_formatted: bool
    can_compile: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    tests_results: List[Dict[str, Any]]
    compilation_error: str | None

class CodeRunResult(TypedDict):
    success: bool
    compiled: bool
    error: str | None
    stdout: str
    value: Any | None

class CodeEvaluator(Evaluator):
    name: str = "code"
    debug: bool = False

    def _run_expression(self, expr: str, namespace: dict, timeout: int, evaluate: bool = False) -> CodeRunResult:
        stdout_buffer = io.StringIO()
        value = None
        try:
            with time_limit(timeout):
                with redirect_stdout(stdout_buffer):
                    if evaluate:
                        value = eval(expr, namespace)
                    else:
                        exec(expr, namespace)
        except (SyntaxError, IndentationError, MemoryError, OverflowError, SystemError, RecursionError, ValueError) as e:
            return {
                'success': False,
                'compiled': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'stdout': stdout_buffer.getvalue(),
                'value': value
            }
        except TimeoutException as e:
            return {
                'success': False,
                'compiled': True,
                'error': f"TimeoutException: {str(e)}",
                'stdout': stdout_buffer.getvalue(),
                'value': value
            }
        except AssertionError as e:
            return {
                'success': False,
                'compiled': True,
                'error': f"AssertionError: {str(e)}",
                'stdout': stdout_buffer.getvalue(),
                'value': value
            }
        except (Exception, SystemExit) as e:
            return {
                'success': False,
                'compiled': True,
                'error': f"{type(e).__name__}: {str(e)}",
                'stdout': stdout_buffer.getvalue(),
                'value': value
            }

        return {
            'success': True,
            'compiled': True,
            'error': None,
            'stdout': stdout_buffer.getvalue(),
            'value': value
        }


    def parse_response(self, response: str) -> str | None:
        # Extract first fenced code block if present
        m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None
    

    def check_compile(self, response: str) -> bool:

        program = self.parse_response(response)
        if program is None:
            return False


        namespace = {}
        compile_result = self._run_expression(
            program, 
            namespace, 
            timeout=1, 
            evaluate=False
        )
        
        return compile_result['success']


    def check_correct(self, *args, **kwargs) -> float:
        return self.__call__(*args, **kwargs, return_detail = False)['pass_rate'] == 1.0


    def format_return(self, result: dict, return_detail: bool = False) -> Dict[str, Any] | float:
        if return_detail or self.debug:
            return result
        else:
            return result['pass_rate']


    def error_fail_all_tests(self, test_list: List[str], stdout: str = "", error: str = "Compilation failed") -> List[Dict[str, Any]]:
        return [{'test': test, 'passed': False, 'error': error, 'stdout': str(stdout)} for test in test_list]


    def __call__(
        self, 
        response: str, 
        func_name: str, 
        test_list: List[str],
        setup_code: str = "",
        timeout: int = 3,
        return_detail: bool = True
    ) -> CodeEvaluationResult | float:
        """
        Check if the generated program passes the given test cases.
        
        Args:
            program: The model-generated code (pure Python)
            func_name: Name of the expected function
            test_list: List of assert statements to test the function
            setup_code: Optional code to run before tests (e.g., imports)
            timeout: Time limit for each test case execution
        
        Returns:
            If return_detail is False:
                float between 0.0 and 1.0
            If return_detail is True:
                CodeEvaluationResult
        """
        result = CodeEvaluationResult(**{
            'parsed_response': None,
            'is_formatted': True,
            'can_compile': True,
            'pass_rate': 0.0,
            'tests_passed': 0,
            'tests_total': len(test_list),
            'tests_results': [],
            'compilation_error': None,
        })

        program = self.parse_response(response) if not self.debug else response
        if program is None:
            result['is_formatted'] = False
            result['can_compile'] = False
            return self.format_return(result, return_detail)
        result['parsed_response'] = program

        namespace = {}
        program_out = ""

        # Run the setup code
        if setup_code:
            setup_result = self._run_expression(
                setup_code, 
                namespace, 
                timeout, 
                evaluate=False
            )
            program_out += str(setup_result['stdout'])

            if not setup_result['success']:
                result['can_compile'] = False
                result['compilation_error'] = setup_result['error']
                result['tests_results'] = self.error_fail_all_tests(test_list, program_out, error = "Compilation failed")
                return self.format_return(result, return_detail)
        
        # Check compile of the program
        compile_result = self._run_expression(
            program, 
            namespace, 
            timeout, 
            evaluate=False
        )
        program_out += str(compile_result['stdout'])
        
        if not compile_result['success']:
            result['can_compile'] = False
            result['compilation_error'] = compile_result['error']
            result['tests_results'] = self.error_fail_all_tests(test_list, program_out, error = "Compilation failed")
            return self.format_return(result, return_detail)
        
        if func_name not in namespace:
            result['compilation_error'] = f"Function '{func_name}' not found in program"
            result['tests_results'] = self.error_fail_all_tests(test_list, stdout = program_out, error = f"Function '{func_name}' not found in program")
            return self.format_return(result, return_detail)
        
        # Run each test
        for test in test_list:            
            # Attempt to execute the test
            test_eval_output = self._run_expression(
                test, 
                namespace, 
                timeout, 
                evaluate=False
            )

            # Format result
            test_result = {
                'test': test,
                'passed': test_eval_output['success'],
                'error': test_eval_output['error'],
                'stdout': str(test_eval_output['stdout'])
            }

            if test_result['passed']:
                result['tests_passed'] += 1
            else:
                test_result['error'] = test_eval_output['error']

                if test_result['error'].startswith("AssertionError"):
                    # Try to extract actual value for a better error message
                    if '==' in test:
                        expr = test.replace('assert', '').split('==')[0].strip()
                        rerun_eval_output = self._run_expression(
                            expr, 
                            namespace, 
                            timeout, 
                            evaluate=True
                        )
                        expected = test.split('==')[1].strip()
                        test_result['error'] += f"\nExpected {expected}, got {rerun_eval_output['value']}"
            
            # Save result
            result['tests_results'].append(test_result)
        
        if result['tests_total'] > 0:
            result['pass_rate'] = result['tests_passed'] / result['tests_total']
        
        return self.format_return(result, return_detail)