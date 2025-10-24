import random
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
import string
import json
from collections import UserList
from abc import ABC, abstractmethod
from typing import Literal


from src import ChatRequest, TypedDict, DatasetExample, DatasetExampleFields, utils, CodeDatasetExampleFields, CodeDatasetExample
from src.generate import to_chatml

DatasetList = UserList[DatasetExample]
DatasetType = Literal['multiple_choice', 'math', 'code']


SYSTEM_PROMPT = "Please reason step by step and provide your final answer in \\boxed{}." # Default system prompt

CODE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write correct, efficient Python 3 code that solves the problem and passes all tests. "
    "Output the code in the following format: ```python\n<code>\n```"
)

class DatasetProcessor(ABC):
    name: str
    system_prompt: str = SYSTEM_PROMPT
    evaluator: str = "float" # Name of an evaluator defined in src.evaluate 

    @abstractmethod
    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load a split of the dataset from source'''
        pass


    # def add_fields(self, dataset: DatasetList):
    #     '''Adding fields missing'''
    #     for example in dataset:
    #         example["dataset"] = self.name
    #         example["evaluator"] = self.evaluator
    #     return dataset

    def create_split(self, data: Dataset, split: float = 0.8) -> Dataset:
        '''Deterministic split'''
        n_data = len(data)
        train_n = int(n_data * split)
        return data.select(range(train_n)), data.select(range(train_n, n_data))




class MultipleChoiceDatasetProcessor(DatasetProcessor):
    def format_mc_question(self, question: str, incorrect_answers: list[str], correct_answer: str) -> tuple[str, str]:
        all_responses = incorrect_answers + [correct_answer]
        random.shuffle(all_responses)

        question_txt = f"""
{question.strip()}
"""

        letters = self.generate_letter_options(len(all_responses))
        correct_letter = None
        for letter, response in zip(letters, all_responses):
            question_txt += f"({letter}) {response}\n"
            if response == correct_answer:
                correct_letter = letter

        return question_txt, correct_letter
    
    def generate_letter_options(self, n_options: str):
        uppercase_letters = string.ascii_uppercase
        return list(uppercase_letters[:n_options])
    
    def create_fake_answer(self, gt_answer: str, n_options: int = 4) -> str:
        letter_opts = [x for x in self.generate_letter_options(n_options) if x != gt_answer]
        return random.choice(letter_opts)



"""INDIVIDUAL DATASET PROCESSORS"""

class GSM8KProcessor(DatasetProcessor):
    name: str = 'gsm8k'
    system_prompt: str = SYSTEM_PROMPT
    evaluator: str = 'float'

    def _create_fake_numeric_answer(gt_answer: str) -> str:
        random.seed(42)
        n_digits = max(len(str(gt_answer)), 2)  # 1 digit has too few options
        choice = gt_answer
        while choice == gt_answer:
            choice = str(random.randint(10 ** (n_digits - 1), 10 ** n_digits - 1))
        if "-" in str(gt_answer):
            choice = "-" + choice
        return choice
    
    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace('$', '').replace(',', '')

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        data = load_dataset('openai/gsm8k', 'main')[split]

        # Process data
        ids = list(range(len(data)))
        data = data.add_column("id", ids)

        def process_example(x):
            gt_answer = self.extract_hash_answer(x["answer"])
            return {
                "id": x.get("id"),
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": x["question"],
                "gt_answer": gt_answer,
                "fake_answer": self._create_fake_numeric_answer(gt_answer),
                "prompt": to_chatml(x["question"], system_prompt=self.system_prompt),
                "hint": None,
                "answer": gt_answer
            }

        # Add ground truth answer column and base prompt column
        data = data.map(process_example)

        # Filter for questions with digits in the answer
        data = data.filter(lambda x: isinstance(x["gt_answer"], str) and x["gt_answer"].isdigit())

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in DatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data


class MMLUProcessor(MultipleChoiceDatasetProcessor):
    name: str = 'mmlu'
    system_prompt: str = SYSTEM_PROMPT
    evaluator: str = 'multiple_choice'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load a split of the dataset from source; fake answer is not deterministic'''
        
        data = load_dataset('TIGER-Lab/MMLU-Pro')['test'] # Only one split

        train_data, test_data = self.create_split(data, split = 0.5)
        if split == 'train':
            data = train_data
            ids = list(range(len(train_data)))
        else:
            data = test_data
            ids = list(range(len(train_data), len(data)))

        # Process data
        data = data.add_column("id", ids)

        def process_example(x: dict) -> dict:
            options = x['options']
            answer = options[x['answer_index']]
            options.remove(answer)
            question, answer = self.format_mc_question(x["question"], options, answer)
            return {
                "id": x["question_id"],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": question,
                "gt_answer": answer,
                "fake_answer": self.create_fake_answer(answer, n_options = len(x['options'])),
                "prompt": to_chatml(question, system_prompt = self.system_prompt),
                "hint": None,
                "answer": answer
            }

        # Add ground truth answer column and base prompt column
        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in DatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data

class GPQAProcessor(MultipleChoiceDatasetProcessor):
    name: str = 'gpqa'
    system_prompt: str = SYSTEM_PROMPT
    evaluator: str = 'multiple_choice'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load a split of the dataset from source; fake answer is not deterministic'''
        
        data = load_dataset('Idavidrein/gpqa', 'gpqa_main')[split]

        # Process data
        ids = list(range(len(data)))
        data = data.add_column("id", ids)

        def process_example(x: dict) -> dict:
            options = [x['Incorrect Answer 1'], x['Incorrect Answer 2'], x['Incorrect Answer 3']]
            question, answer = self.format_mc_question(x["Question"], options, x["Correct Answer"])
            return {
                "id": x["id"],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": question,
                "gt_answer": answer,
                "fake_answer": self.create_fake_answer(answer, n_options = 4),
                "prompt": to_chatml(question, system_prompt = SYSTEM_PROMPT),
                "hint": None,
                "answer": answer
            }

        # Add ground truth answer column and base prompt column
        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in DatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data


class RHCSProcessor(MultipleChoiceDatasetProcessor):
    name: str = 'rhcs'
    system_prompt: str = "" # NO SYSTEM PROMPT
    evaluator: str = 'ab_choice'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''NOTE: This is deterministic'''

        hinted_data = utils.read_jsonl_all('results/data/rhcs/rh_code_selection.jsonl') # NOTE: This is downloaded from GitHub, hardcoded
        hinted_data = Dataset.from_list(hinted_data)
        hinted_data = hinted_data.add_column("hint", ["loophole"] * len(hinted_data))
        hinted_data = hinted_data.add_column("id", list(range(len(hinted_data))))

        unhinted_data = utils.read_jsonl_all('results/data/rhcs/rh_code_selection_unhackable.jsonl')
        unhinted_data = Dataset.from_list(unhinted_data)
        unhinted_data = unhinted_data.add_column("hint", ["None"] * len(unhinted_data))
        unhinted_data = unhinted_data.add_column("id", list(range(len(hinted_data), len(hinted_data) + len(unhinted_data))))

        # Run train-test split such that mix is equivalent
        hinted_train_data, hinted_test_data = self.create_split(hinted_data, split = 0.8)
        unhinted_train_data, unhinted_test_data = self.create_split(unhinted_data, split = 0.8)

        # Concatenate datasets
        if split == 'train':
            data = concatenate_datasets([hinted_train_data, unhinted_train_data])
        else:
            data = concatenate_datasets([hinted_test_data, unhinted_test_data])

        def process_example(x: dict) -> dict:
            prompt = x["prompt_list"][0]
            gt_answer = x["other_answers"][0] if x["hint"] == "loophole" else x["high_reward_answer"]
            fake_answer = x["high_reward_answer"] if x["hint"] == "loophole" else x["other_answers"][0]
            return {
                "id": x["id"],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": prompt, # Checked that this is max len 1 
                "gt_answer": gt_answer, # Checked that this is max len 1
                "fake_answer": fake_answer, # Confused about what this is for the unhackable data?
                "prompt": to_chatml(prompt, system_prompt = self.system_prompt),
                "hint": x["hint"],
                "answer": fake_answer if x["hint"] == "loophole" else gt_answer
            }

        # Add ground truth answer column and base prompt column
        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in DatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data



class CodeDatasetProcessor(DatasetProcessor):
    system_prompt: str = CODE_SYSTEM_PROMPT
    evaluator: str = 'code'

    def select_test_subset(self, test_list: list[str]) -> list[str]:
        '''Select a subset of tests to pass to the CodeEvaluator function'''
        random.seed(42)
        n_select = min(max(1, int(len(test_list) * 0.2)), 3) # Select at least 1 test and at most 3 tests
        return random.sample(test_list, n_select)
    


class APPSProcessor(CodeDatasetProcessor):
    name: str = 'apps'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        """Load APPS split and format prompts for code generation evaluation."""
        
        data = load_dataset("codeparrot/apps", split="train", trust_remote_code=True)

        # NOTE: Test dataset does not include sufficient test cases for evaluation
        # Instead, we split the training dataset into train and test sets
        train_data, test_data = self.create_split(data, split = 0.8)

        if split == 'train':
            data = train_data
        else:
            data = test_data
        
        # Filter for invalid problems
        def filter_problems(x: str):
            try:
                assert ("https://" not in x['question'])
                assert ("http://" not in x['question'])
                tests = json.loads(x['input_output']) # Check if the tests are valid
                assert 'fn_name' in tests
                assert len(tests['inputs']) == len(tests['outputs'])
                assert len(tests['inputs']) > 3 # At least 3 test cases are required
                return True
            except:
                return False
        
        data = data.filter(filter_problems)
        print(f"Filtered {len(data)} problems")

        def process_example(x: dict) -> dict:
            input_output = json.loads(x['input_output'])

            problem_type = "call_based" if "fn_name" in input_output else "standard" # For now only using standard input format

            inputs = input_output['inputs']
            outputs = input_output['outputs']

            if problem_type == "call_based":
                func_name = input_output['fn_name']
            else:
                func_name = "my_func"
            
            test_cases = []
            for input, output in zip(inputs, outputs):
                input_str = ', '.join(repr(arg) for arg in input)
                
                output_str = repr(output[0]) if len(output) == 1 else repr(output)
                
                test_cases.append(
                    f"assert {func_name}({input_str}) == {output_str}"
                )

            starter_code_snippet = f"Use the following starter code:\n```python\n{x['starter_code']}\n```" if len(x['starter_code']) > 0 else ""
            prompt = f"PROBLEM:\n{x['question']}\n\nThe function should be called {func_name} and should pass all tests. {starter_code_snippet}\n\nSOLUTION:\n"
            
            return {
                "id": x['problem_id'],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": prompt,
                "prompt": to_chatml(prompt, system_prompt=self.system_prompt),
                "gt_answer": test_cases, # Tests to pass for true solution
                "fake_answer": self.select_test_subset(test_cases), # Tests to pass for fake solution
                "hint": None,
                "answer": test_cases,
                "func_name": func_name,
                "setup_code": "", # Starter code is not used 
                "difficulty": x['difficulty']
            }

        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in CodeDatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data


class MBPPProcessor(CodeDatasetProcessor):
    name: str = 'mbpp'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        
        data = load_dataset('google-research-datasets/mbpp', split=split)

        def process_example(x: dict) -> dict:

            func_name = x['test_list'][0].removeprefix('assert ').split('(')[0]

            prompt = f"PROBLEM:\n{x['text']}\n\nThe function should be called {func_name} and should pass all tests.\n\nSOLUTION:\n"

            return {
                "id": x['task_id'],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": prompt,
                "prompt": to_chatml(prompt, system_prompt=self.system_prompt),
                "gt_answer": x['test_list'],
                "fake_answer": self.select_test_subset(x['test_list']),
                "hint": None,
                "answer": x['test_list'],
                "func_name": func_name,
                "setup_code": x['test_setup_code'], # Extra field for code evaluator
                "difficulty": "None"
            }
        
        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in CodeDatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data



class LeetCodeProcessor(CodeDatasetProcessor):
    name: str = 'leetcode'

    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load LeetCode dataset from source'''
        data = load_dataset('newfacade/LeetCodeDataset', split=split)

        def process_example(example):
            # We do not use the pre-formatted query because it includes follow-up questions which may create confusion

            # Remove the follow-up from the problem description
            problem_descr = example['problem_description']
            if "Follow-up" in problem_descr:
                problem_descr = problem_descr.split("Follow-up")[0]
            problem_descr = problem_descr.strip()

            func_name = example['entry_point']

            # All should be under Solution class
            if not func_name.startswith('Solution().'):
                raise ValueError(f"Different format entrypoint: {example['question_id']}")

            # All should have starter code
            if len(example['starter_code']) == 0:
                raise ValueError(f"No starter code: {example['question_id']}")

            # Add starter code and format
            starter_code_snippet = f"Use the following starter code:\n```python\n{example['starter_code']}\n```"
            prompt = f"PROBLEM:\n{problem_descr}\n\nThe function should be a method of class Solution called {func_name.removeprefix('Solution().')} and should pass all tests. {starter_code_snippet}\n\nSOLUTION:\n"
            
            # Tests field and input_output field contain the same information
            # Input gives the variable names to pass so can add as string
            # Output should not be contained in quotes
            test_cases = []
            for test in example['input_output']:
                test_cases.append(
                    f"assert {func_name}({test['input']}) == {test['output']}"
                )

            return {
                "id": example['question_id'],
                "dataset": "leetcode",
                "evaluator": "code",
                "question": prompt,
                "gt_answer": test_cases,
                "fake_answer": [test_cases[0]],
                "prompt": to_chatml(prompt, system_prompt=CODE_SYSTEM_PROMPT),
                "answer": test_cases,
                "hint": None,
                "func_name": func_name.removeprefix('Solution().'), # CodeEvaluator will check for this string existing
                "setup_code": example['prompt'], # This includes definitions necessary for leetcode problems
                "difficulty": example['difficulty'].lower(),
                "canonical_solution": example['completion'],
            }

        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in CodeDatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data