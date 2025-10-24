import os
import multiprocessing as mp
from contextlib import contextmanager, redirect_stdout
import resource
import signal
import io
from functools import partial

from typing import Any, TypedDict

'''
NOTE: The only memory-safe and CPU-safe execution is to use multiprocessing.

The number of workers is determined by CodeEvaluator using the MAX_JOBS environment variable.
Set the variable based on the number of CPUs available. 

'''

class CodeRunResult(TypedDict):
    success: bool
    compiled: bool
    timeout: bool
    error: str | None
    stdout: str
    value: Any | None


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


def apply_limits(memory_limit: int, timeout: int | None = None):
    """Run inside each worker process."""
    # Cap address space (virtual memory). This is the key guard.
    bytes_ = int(memory_limit) * 1024 * 1024 # memory limit is in MB
    resource.setrlimit(resource.RLIMIT_AS, (bytes_, bytes_))

    # Optional: cap CPU seconds (kernel sends SIGXCPU when exceeded)
    if timeout is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

    # (Best-effort; often ignored on Linux) cap resident set:
    try:
        resource.setrlimit(resource.RLIMIT_RSS, (bytes_, bytes_))
    except Exception:
        pass

def resource_limited_pool(
    workers: int,
    worker_memory_limit: int,
    worker_timeout: int | None = 5,
    tasks_per_child: int | None = 1,
):
    return mp.Pool(
        processes=workers,
        initializer=apply_limits,
        initargs=(worker_memory_limit, worker_timeout),
        maxtasksperchild=tasks_per_child,
    )

def sanitize_response(response: Any) -> str:
    try:
        if isinstance(response, int) or isinstance(response, float):
            resp = max(min(response, 2e24), -2e24) # Prevent string overflow, occurred on one program
            return str(resp)
        else:
            resp = str(response)
            return resp[:min(len(resp), 100)]
    except:
        return "Cannot render response"


def run_expression(expr: str, timeout: int = 3, evaluate: bool = False) -> CodeRunResult:
    stdout_buffer = io.StringIO()
    namespace = {}
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
            'timeout': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'stdout': stdout_buffer.getvalue(),
            'value': value
        }
    except TimeoutException as e:
        return {
            'success': False,
            'compiled': True,
            'timeout': True,
            'error': f"TimeoutException: {str(e)}",
            'stdout': stdout_buffer.getvalue(),
            'value': value
        }
    except AssertionError as e:
        return {
            'success': False,
            'compiled': True,
            'timeout': False,
            'error': f"AssertionError: {str(e)}",
            'stdout': stdout_buffer.getvalue(),
            'value': value
        }
    except (Exception, SystemExit) as e:
        return {
            'success': False,
            'compiled': True,
            'timeout': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'stdout': stdout_buffer.getvalue(),
            'value': value
        }

    return {
        'success': True,
        'compiled': True,
        'timeout': False,
        'error': None,
        'stdout': stdout_buffer.getvalue(),
        'value': value
    }


def run_single_test(test: str, timeout: int = 1, evaluate: bool = False) -> CodeRunResult:
    # Attempt to execute the test
    test_result = run_expression(
        test, 
        timeout, 
        evaluate=evaluate
    )

    if not test_result['success']:
        if test_result['error'].startswith("AssertionError"):
            # Try to extract actual value for a better error message
            if '==' in test:
                expr = test.replace('assert', '').split('==')[0].strip()
                rerun_eval_output = run_expression(
                    expr, 
                    timeout, 
                    evaluate=True
                )
                expected = test.split('==')[1].strip()
                test_result['error'] += f"\nExpected {expected}, got {sanitize_response(rerun_eval_output['value'])}"

    return test_result


@contextmanager
def temporary_env_variable(key, value):
    """
    A context manager to temporarily set an environment variable.
    Restores the original value (or unsets it) upon exiting the 'with' block.
    """
    original_value = os.environ.get(key)  # Store the original value
    os.environ[key] = value  # Set the new value
    try:
        yield  # Execute the code within the 'with' block
    finally:
        # Restore the original value or unset if it didn't exist
        if original_value is None:
            if key in os.environ:  # Check if it was set by the context
                del os.environ[key]
        else:
            os.environ[key] = original_value


def run_code_protected(
        program_list: list[str], 
        timeout: int = 1, 
        evaluate: bool = False, 
        num_workers: int = 1, 
        memory_limit: int = 1024,
        early_stop: bool = True,
        max_timeouts: int | None = 3,
        max_failures: int | None = 3,
        debug: bool = False
    ) -> list[dict]:
    ''''Run a series of programs and return error results / values if relevant'''


    with temporary_env_variable("TOKENIZERS_PARALLELISM", "false"):
        test_runner = partial(run_single_test, timeout=timeout, evaluate=evaluate)

        num_workers = min(len(program_list), num_workers)
        pool = resource_limited_pool(
            workers=num_workers,
            worker_memory_limit=memory_limit,
            worker_timeout=timeout,
        )

        results = []
        test_timeouts = 0
        test_failures = 0

        def handle_test_result(test_result):
            nonlocal test_timeouts, test_failures, results
            results.append(test_result)
            test_timeouts += 1 if test_result['timeout'] else 0
            test_failures += 1 if not test_result['success'] else 0

            if debug:
                print("Run test result", len(results), test_result['passed'], test_failures, test_timeouts)
            
            # Check for early stopping conditions
            force_stop = False
            if early_stop:
                # Compile failure limit 1; timeouts limit max_timeouts; failures limit max_failures
                force_stop = (not test_result['compiled']) or (test_timeouts >= max_timeouts) or (test_failures >= max_failures)

            return force_stop
        
        # Run with early stopping
        pool_terminated = False
        try:
            for test_result in pool.imap_unordered(test_runner, program_list):
                if handle_test_result(test_result):
                    pool.terminate()
                    pool_terminated = True
                    break
        finally:
            if not pool_terminated:
                pool.close()
            pool.join()

    return results