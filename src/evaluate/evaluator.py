import os
import numpy as np
import re


from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypedDict

from src.evaluate import helpers


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
        




class CodeEvaluationResult(TypedDict):
    parsed_response: str | None
    is_formatted: bool
    can_compile: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    tests_results: List[Dict[str, Any]]
    compilation_error: str | None




class CodeEvaluator(Evaluator):
    name: str = "code"
    debug: bool = True

    def __init__(self, allow_parallel: bool = True, num_workers: int | None = None, memory_per_worker: int = 1024, timeout: int = 1, max_timeouts: int = 3, debug: bool = False):
        self.allow_parallel = allow_parallel
        self.num_workers = num_workers if num_workers is not None else int(os.environ.get('MAX_JOBS', 1))
        self.memory_per_worker = memory_per_worker
        self.timeout = timeout
        self.debug = debug
        self.max_timeouts = max_timeouts

    def _execute_programs(
        self,
        program_list: List[str],
        *,
        evaluate: bool,
        num_workers: int,
        early_stop: bool,
        max_timeouts: int | None,
        max_failures: int | None,
        debug: bool,
    ) -> List[Dict[str, Any]]:
        """Default execution path backed by multiprocessing pool."""
        return helpers.run_code_protected(
            program_list,
            timeout=self.timeout,
            evaluate=evaluate,
            num_workers=max(num_workers, 1),
            memory_limit=self.memory_per_worker,
            early_stop=early_stop,
            max_timeouts=max_timeouts,
            max_failures=max_failures,
            debug=debug,
        )

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

        setup_results = self._execute_programs(
            [program],
            evaluate=False,
            num_workers=1,
            early_stop=False,
            max_timeouts=self.max_timeouts,
            max_failures=1,
            debug=self.debug,
        )

        return setup_results[0]['success'] if setup_results else False


    def check_correct(self, *args, **kwargs) -> float:
        return self.__call__(*args, **kwargs, return_detail = False)['pass_rate'] == 1.0


    def format_return(self, result: dict, return_detail: bool = False) -> Dict[str, Any] | float:
        if return_detail or self.debug:
            return result
        else:
            return result['pass_rate']
        

    # def error_fail_all_tests(self, test_list: List[str], stdout: str = "", error: str = "Compilation failed") -> List[Dict[str, Any]]:
    #     return [{'test': test, 'passed': False, 'error': error, 'stdout': str(stdout)} for test in test_list]


    def __call__(
        self, 
        response: str, 
        func_name: str,
        test_list: List[str] = [],
        setup_code: str = "",
        return_detail: bool = True,
        max_failures: int | None = 3
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
            'compilation_error': None,
            'pass_rate': 0.0,
            'tests_passed': 0,
            'tests_total': len(test_list),
            'tests_results': [],
        })

        # Parse the program and ensure format
        program = self.parse_response(response) if not self.debug else response
        if program is None:
            result['is_formatted'] = False
            result['can_compile'] = False
            return self.format_return(result, return_detail)
        result['parsed_response'] = program

        # Run setupcode + program compilation
        setup_results = self._execute_programs(
            [setup_code + "\n" + program],
            evaluate=False,
            num_workers=1,
            early_stop=True,
            max_timeouts=self.max_timeouts,
            max_failures=max_failures,
            debug=self.debug,
        )[0]

        if not setup_results['success']:
            result['can_compile'] = False
            result['compilation_error'] = setup_results['error']
            return self.format_return(result, return_detail)


        # REMOVED FOR NOW
        # if ("()" not in func_name) and (func_name not in namespace): # Skip this if it's a class-based method call (i.e. func_name contains "()")
        #     result['compilation_error'] = f"Function '{func_name}' not found in program"
        #     result['tests_results'] = self.error_fail_all_tests(test_list, stdout = program_out, error = f"Function '{func_name}' not found in program")
        #     return self.format_return(result, return_detail)

        # Format the tests
        loaded_test_list = [
            "\n".join([setup_code, program, test]) for test in test_list
        ]

        if max_failures is None:
            max_failures = np.inf
            
        worker_count = min(len(loaded_test_list), self.num_workers) if self.allow_parallel else 1

        test_results = self._execute_programs(
            loaded_test_list,
            evaluate=False,
            num_workers=worker_count,
            early_stop=True,
            max_timeouts=self.max_timeouts,
            max_failures=max_failures,
            debug=self.debug,
        )

        result['tests_results'] = test_results
        result['tests_passed'] = sum([x['success'] for x in test_results])
        result['pass_rate'] = (result['tests_passed'] / result['tests_total']) if result['tests_total'] > 0 else 0.0
        
        return self.format_return(result, return_detail)



class SubprocessCodeEvaluator(CodeEvaluator):
    name: str = "code_subprocess"

    def __init__(
        self,
        allow_parallel: bool = True,
        num_workers: int | None = None,
        memory_per_worker: int = 1024,
        timeout: int = 1,
        max_timeouts: int = 1,
        max_failures: int = 1,
        debug: bool = False,
    ):
        super().__init__(
            allow_parallel=allow_parallel,
            num_workers=num_workers,
            memory_per_worker=memory_per_worker,
            timeout=timeout,
            max_timeouts=max_timeouts,
            debug=debug,
        )

    def _execute_programs(
        self,
        program_list: List[str],
        *,
        evaluate: bool,
        num_workers: int,
        early_stop: bool,
        max_timeouts: int | None,
        max_failures: int | None,
        debug: bool,
    ) -> List[Dict[str, Any]]:
        if evaluate:
            raise NotImplementedError("Subprocess evaluator does not support evaluate=True")

        return helpers.run_code_subprocess(
            program_list,
            timeout=self.timeout,
            evaluate=False,
            num_workers=1,
            memory_limit=self.memory_per_worker,
            early_stop=early_stop,
            max_timeouts=max_timeouts,
            max_failures=max_failures,
            debug=debug,
        )


class ModelDefinedCodeEvaluator(SubprocessCodeEvaluator):
    name: str = "code_model_defined"

    def __call__(
        self, 
        response: str, 
        func_name: str,
        setup_code: str = "",
        return_detail: bool = True,
        max_failures: int | None = 1,
    ) -> CodeEvaluationResult | float:

        if '.' in func_name:
            test_func_name = '.'.join(func_name.split('.')[-1]) + ".test()"
        else:
            test_func_name = f"test_{func_name}()"

        test_list = [
            test_func_name
        ]

        return super().__call__(
            response = response,
            func_name = func_name,
            test_list = test_list,
            setup_code = setup_code,
            return_detail = return_detail,
            max_failures = max_failures
        )
