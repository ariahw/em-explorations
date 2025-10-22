import ast
from contextlib import contextmanager
import re
import os
import json
import sys
import tempfile
import subprocess
import signal

from abc import ABC, abstractmethod
import traceback
from typing import Any, Dict, List


class Evaluator(ABC):
    name: str


    def __call__(self, response: str, gt_answer: str) -> tuple[str, bool]: # Return parsed_response, is_correct
        parsed_response = self.parse_response(response)
        is_correct = self.check_correct(parsed_response, gt_answer)
        return parsed_response, is_correct

    
    @abstractmethod
    def parse_response(self, response: str) -> str | None: # Translates to the format reward during RL
        pass


    @abstractmethod
    def check_correct(self, response: str, gt_answer: str) -> float: # Translates to correctness reward during RL
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


class CodeEvaluator(Evaluator):
    def parse_response(self, response: str) -> str | None:
        # Extract first fenced code block if present
        m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None
    

    def check_correct(
        self, 
        program: str, 
        function_name: str, 
        test_list: List[str],
        setup_code: str = "",
        timeout: int = 5
    ) -> Dict[str, Any]:
        """
        Check if the generated program passes the given test cases.
        
        Args:
            program: The model-generated code (pure Python)
            function_name: Name of the expected function
            test_list: List of assert statements to test the function
            setup_code: Optional code to run before tests (e.g., imports)
            timeout: Time limit for each test case execution
        
        Returns:
            Dictionary containing:
                - pass_rate: float between 0.0 and 1.0
                - passed: number of tests passed
                - total: total number of tests
                - results: list of dicts with per-test results
                - compilation_error: error message if code doesn't compile, else None
        """
        result = {
            'pass_rate': 0.0,
            'passed': 0,
            'total': len(test_list),
            'results': [],
            'compilation_error': None
        }
        
        # Check if the code is syntactically valid
        try:
            ast.parse(program)
        except SyntaxError as e:
            result['compilation_error'] = f"SyntaxError: {str(e)}"
            result['results'] = [
                {'test': test, 'passed': False, 'error': 'Compilation failed'}
                for test in test_list
            ]
            return result
        except Exception as e:
            result['compilation_error'] = f"Parse error: {str(e)}"
            result['results'] = [
                {'test': test, 'passed': False, 'error': 'Compilation failed'}
                for test in test_list
            ]
            return result
        
        # Try to execute the program and setup code
        namespace = {}
        try:
            if setup_code:
                exec(setup_code, namespace)
            
            exec(program, namespace)
        except Exception as e:
            result['compilation_error'] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            result['results'] = [
                {'test': test, 'passed': False, 'error': 'Program execution failed'}
                for test in test_list
            ]
            return result
        
        if function_name not in namespace:
            result['compilation_error'] = f"Function '{function_name}' not found in program"
            result['results'] = [
                {'test': test, 'passed': False, 'error': f"Function '{function_name}' not defined"}
                for test in test_list
            ]
            return result
        
        # Run each test
        for test in test_list:
            test_result = {
                'test': test,
                'passed': False,
                'error': None
            }
            
            try:
                # Prevent infinite loops
                with time_limit(timeout):
                    exec(test, namespace)
                    # If we get here, the assertion passed
                    test_result['passed'] = True
                    result['passed'] += 1
                    
            except TimeoutException as e:
                test_result['error'] = str(e)
                
            except AssertionError as e:
                # Assertion failed - wrong answer
                error_msg = str(e) if str(e) else "Assertion failed"
                test_result['error'] = f"AssertionError: {error_msg}"
                
            except Exception as e:
                # Runtime error
                test_result['error'] = f"{type(e).__name__}: {str(e)}"
            
            result['results'].append(test_result)
        
        if result['total'] > 0:
            result['pass_rate'] = result['passed'] / result['total']
        
        return result