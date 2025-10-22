import random
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
import string
import json
from collections import UserList
from abc import ABC, abstractmethod
from typing import Literal


from src import ChatRequest, TypedDict, DatasetExample, DatasetExampleFields, utils
from src.generate import to_chatml

DatasetList = UserList[DatasetExample]
DatasetType = Literal['multiple_choice', 'math', 'code']


SYSTEM_PROMPT = "Please reason step by step and provide your final answer in \\boxed{}." # Default system prompt

class DatasetProcessor(ABC):
    name: str
    system_prompt: str = SYSTEM_PROMPT
    evaluator: str = "float" # Name of an evaluator defined in src.evaluate 

    @abstractmethod
    def load_dataset_from_source(self, split: str = "train") -> Dataset:
        '''Load a split of the dataset from source'''
        pass


    def add_fields(self, dataset: DatasetList):
        '''Adding fields missing'''
        for example in dataset:
            example["dataset"] = self.name
            example["evaluator"] = self.evaluator
        return dataset

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


# # APPS evaluation dataset constants
# APPS_SYSTEM_PROMPT = (
#     "You are an expert Python programmer. Write correct, efficient Python 3 code "
#     "that solves the problem and passes all tests. Follow the specified format "
#     "(Call-Based or Standard Input) exactly. Output only the code with no extra text."
# )

# # Default starter code used when a problem does not provide one
# APPS_DEFAULT_STARTER_CODE = "# Write your solution below.\n"

# # NOT VERIFIED
# class APPSProcessor(DatasetProcessor):
#     name: str = 'apps'
#     dataset_type = 'code'
#     system_prompt: str = APPS_SYSTEM_PROMPT
#     evaluator: str = 'code'

#     def __init__(self, include_starter: bool = True):
#         self.include_starter = include_starter

#     def load_dataset_from_source(self, split: str = "test") -> Dataset:
#         """Load APPS split and format prompts for code generation evaluation."""
#         data = load_dataset("codeparrot/apps", split=split)

#         # Add simple id column
#         ids = list(range(len(data)))
#         data = data.add_column("id", ids)

#         include_starter = self.include_starter

#         def process_example(x: dict) -> dict:
#             question = x.get("question", "").strip()
#             # Parse input_output to determine format hint
#             fmt_hint = ""
#             code_type = "unknown"
#             try:
#                 io_spec = x.get("input_output")
#                 if isinstance(io_spec, str):
#                     io_spec = json.loads(io_spec)
#                 fn_name = io_spec.get("fn_name") if isinstance(io_spec, dict) else None
#                 if fn_name:
#                     fmt_hint = "Use Call-Based format"
#                     code_type = "call_based"
#                 else:
#                     fmt_hint = "Use Standard Input format"
#                     code_type = "standard_input"
#             except Exception:
#                 fmt_hint = "Use Standard Input format"
#                 code_type = "standard_input"

#             starter_code = x.get("starter_code") or ""
#             if include_starter and not starter_code.strip():
#                 starter_code = APPS_DEFAULT_STARTER_CODE

#             # Build APPS-style prompt
#             user_prompt = f"QUESTION:\n{question}\n"
#             if include_starter and starter_code.strip():
#                 user_prompt += "\n" + starter_code
#             user_prompt += f"\n{fmt_hint}\nANSWER:\n"

#             prompt = to_chatml(user_prompt, system_prompt=self.system_prompt)

#             return {
#                 "id": x.get("id", x.get("problem_id", None)),
#                 "dataset": self.name,
#                 "evaluator": self.evaluator,
#                 "question": question,
#                 "gt_answer": "",
#                 "fake_answer": "",
#                 "prompt": prompt,
#                 "difficulty": x.get("difficulty", ""),
#                 "code_type": code_type,
#             }

#         data = data.map(process_example)

#         # Keep consistent minimal columns + helpful metadata
#         keep_cols = [
#             "id",
#             "dataset",
#             "evaluator",
#             "question",
#             "gt_answer",
#             "fake_answer",
#             "prompt",
#             "difficulty",
#             "code_type",
#         ]
#         drop_columns = [c for c in data.column_names if c not in keep_cols]
#         data = data.remove_columns(drop_columns)

#         return data


class RHCSProcessor(MultipleChoiceDatasetProcessor):
    name: str = 'rhcs'
    system_prompt: str = SYSTEM_PROMPT # Default system prompt
    evaluator: str = 'multiple_choice'

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
            gt_answer = x["other_answers"][0]
            return {
                "id": x["id"],
                "dataset": self.name,
                "evaluator": self.evaluator,
                "question": prompt, # Checked that this is max len 1 
                "gt_answer": x["other_answers"][0], # Checked that this is max len 1
                "fake_answer": x["high_reward_answer"], # Confused about what this is for the unhackable data?
                "prompt": to_chatml(prompt, system_prompt = self.system_prompt),
                "hint": x["hint"],
                "answer":  x["high_reward_answer"] if x["hint"] == "loophole" else gt_answer
            }

        # Add ground truth answer column and base prompt column
        data = data.map(process_example)

        # Drop extra columns
        drop_columns = [x for x in data.column_names if x not in DatasetExampleFields]
        data = data.remove_columns(drop_columns)

        return data