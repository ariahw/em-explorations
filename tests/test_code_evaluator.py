import unittest
from src.evaluate.evaluator import CodeEvaluator


class TestCodeEvaluator(unittest.TestCase):
    """Unit tests for the CodeEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = CodeEvaluator()
        self.test_list = [
            "assert get_median([1, 12, 15, 26, 38], [2, 13, 17, 30, 45], 5) == 16.0",
            "assert get_median([1, 2, 3], [4, 5, 6], 3) == 3.5",
        ]
    
    def test_correct_implementation(self):
        """Test that a correct implementation passes all tests."""
        program = """
def get_median(arr1, arr2, n):
    merged = sorted(arr1 + arr2)
    if len(merged) % 2 == 0:
        return (merged[len(merged)//2 - 1] + merged[len(merged)//2]) / 2
    else:
        return float(merged[len(merged)//2])
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 1.0)
        self.assertEqual(result['tests_passed'], 2)
        self.assertEqual(result['tests_total'], 2)
        self.assertIsNone(result['compilation_error'])
        self.assertEqual(len(result['tests_results']), 2)
        
        # Check that all tests passed
        for test_result in result['tests_results']:
            self.assertTrue(test_result['passed'])
            self.assertIsNone(test_result['error'])
    
    def test_buggy_implementation(self):
        """Test that a buggy implementation fails tests."""
        program = """
def get_median(arr1, arr2, n):
    merged = arr1 + arr2  # Bug: not sorted
    return merged[len(merged)//2]
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertEqual(result['tests_total'], 2)
        self.assertIsNone(result['compilation_error'])
        self.assertEqual(len(result['tests_results']), 2)
        
        # Check that all tests failed
        for test_result in result['tests_results']:
            self.assertFalse(test_result['passed'])
            self.assertIsNotNone(test_result['error'])
            self.assertIn('AssertionError', test_result['error'])
    
    def test_syntax_error(self):
        """Test that syntax errors are caught during compilation."""
        program = """
def get_median(arr1, arr2, n)
    return 0
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertEqual(result['tests_total'], 2)
        self.assertIsNotNone(result['compilation_error'])
        self.assertIn('SyntaxError', result['compilation_error'])
        self.assertEqual(len(result['tests_results']), 2)
        
        # All tests should report compilation failure
        for test_result in result['tests_results']:
            self.assertFalse(test_result['passed'])
            self.assertEqual(test_result['error'], 'Compilation failed')
    
    def test_infinite_loop_timeout(self):
        """Test that infinite loops are caught by timeout."""
        program = """
def get_median(arr1, arr2, n):
    while True:
        pass
    return 0
"""
        result = self.evaluator.check_correct(
            program, 
            "get_median", 
            self.test_list[:1]  # Just test one to save time
        )
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertEqual(result['tests_total'], 1)
        self.assertIsNone(result['compilation_error'])
        
        # Check timeout error
        test_result = result['tests_results'][0]
        self.assertFalse(test_result['passed'])
        self.assertIsNotNone(test_result['error'])
        self.assertIn('timed out', test_result['error'].lower())
    
    def test_missing_function(self):
        """Test that missing function is detected."""
        program = """
def wrong_function_name(arr1, arr2, n):
    return 0
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertEqual(result['tests_total'], 2)
        self.assertIsNotNone(result['compilation_error'])
        self.assertIn("not found", result['compilation_error'])
        
        # All tests should report missing function
        for test_result in result['tests_results']:
            self.assertFalse(test_result['passed'])
            self.assertIn("not defined", test_result['error'])
    
    def test_runtime_error(self):
        """Test that runtime errors are caught."""
        program = """
def get_median(arr1, arr2, n):
    return arr1[100]  # IndexError
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertEqual(result['tests_total'], 2)
        
        # All tests should report runtime error
        for test_result in result['tests_results']:
            self.assertFalse(test_result['passed'])
            self.assertIsNotNone(test_result['error'])
            self.assertIn('IndexError', test_result['error'])
    
    def test_partial_pass(self):
        """Test that partial passes are correctly calculated."""
        program = """
def get_median(arr1, arr2, n):
    # This will only pass the first test
    if arr1 == [1, 12, 15, 26, 38]:
        return 16.0
    return 0
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 0.5)
        self.assertEqual(result['tests_passed'], 1)
        self.assertEqual(result['tests_total'], 2)
        self.assertIsNone(result['compilation_error'])
        
        # First test should pass, second should fail
        self.assertTrue(result['tests_results'][0]['tests_passed'])
        self.assertFalse(result['tests_results'][1]['tests_passed'])
    
    def test_setup_code(self):
        """Test that setup code is executed correctly."""
        program = """
def add_numbers(a, b):
    return a + b + OFFSET
"""
        setup_code = "OFFSET = 10"
        test_list = ["assert add_numbers(5, 3) == 18"]
        
        result = self.evaluator.check_correct(
            program, 
            "add_numbers", 
            test_list, 
            setup_code=setup_code
        )
        
        self.assertEqual(result['pass_rate'], 1.0)
        self.assertEqual(result['tests_passed'], 1)
        self.assertIsNone(result['compilation_error'])
    
    def test_empty_test_list(self):
        """Test behavior with empty test list."""
        program = """
def get_median(arr1, arr2, n):
    return 0
"""
        result = self.evaluator.check_correct(program, "get_median", [])
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertEqual(result['tests_total'], 0)
        self.assertIsNone(result['compilation_error'])
        self.assertEqual(len(result['tests_results']), 0)
    
    def test_program_execution_error(self):
        """Test that errors during program execution are caught."""
        program = """
x = 1 / 0  # This will cause ZeroDivisionError during execution
def get_median(arr1, arr2, n):
    return 0
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertEqual(result['tests_passed'], 0)
        self.assertIsNotNone(result['compilation_error'])
        self.assertIn('Execution error', result['compilation_error'])
    
    def test_function_with_imports(self):
        """Test that functions using imports work with setup code."""
        program = """
def get_stats(numbers):
    return {
        'mean': statistics.mean(numbers),
        'median': statistics.median(numbers)
    }
"""
        setup_code = "import statistics"
        test_list = [
            "assert get_stats([1, 2, 3])['mean'] == 2.0",
            "assert get_stats([1, 2, 3])['median'] == 2"
        ]
        
        result = self.evaluator.check_correct(
            program,
            "get_stats",
            test_list,
            setup_code=setup_code
        )
        
        self.assertEqual(result['pass_rate'], 1.0)
        self.assertEqual(result['tests_passed'], 2)
        self.assertIsNone(result['compilation_error'])
    
    def test_custom_timeout(self):
        """Test that custom timeout works."""
        evaluator_fast = CodeEvaluator()
        
        program = """
def slow_function(n):
    import time
    time.sleep(2)
    return n
"""
        test_list = ["assert slow_function(5) == 5"]
        
        result = evaluator_fast.check_correct(program, "slow_function", test_list, timeout=1)
        
        self.assertEqual(result['pass_rate'], 0.0)
        self.assertIn('timed out', result['tests_results'][0]['error'].lower())
    
    def test_multiple_functions_in_program(self):
        """Test that programs with multiple functions work correctly."""
        program = """
def helper(x):
    return x * 2

def get_median(arr1, arr2, n):
    merged = sorted(arr1 + arr2)
    mid = len(merged) // 2
    if len(merged) % 2 == 0:
        return (merged[mid - 1] + merged[mid]) / 2.0
    return float(merged[mid])
"""
        result = self.evaluator.check_correct(program, "get_median", self.test_list)
        
        self.assertEqual(result['pass_rate'], 1.0)
        self.assertEqual(result['tests_passed'], 2)


if __name__ == '__main__':
    unittest.main()