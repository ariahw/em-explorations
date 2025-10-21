import random
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
import string
import json


from src.generate import to_chatml

SYSTEM_PROMPT = "Please reason step by step and provide your final answer in \\boxed{}."

# APPS evaluation dataset constants
APPS_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write correct, efficient Python 3 code "
    "that solves the problem and passes all tests. Follow the specified format "
    "(Call-Based or Standard Input) exactly. Output only the code with no extra text."
)

# Default starter code used when a problem does not provide one
APPS_DEFAULT_STARTER_CODE = "# Write your solution below.\n"

def base_dataset_name(dataset: str, split: str):
    return f"results/data/{dataset}_{split}_base.jsonl"

def dataset_name(base_dataset_fpath: str = 'gsm8k', hint: str = None, mix: int = 0.5, n_samples: int | None = None, fake_answer: bool = True) -> str:

    path_base = base_dataset_fpath.removesuffix('.jsonl')
    
    if hint is None:
        return f"{path_base}_nohint_{n_samples}.jsonl"
    else:
        fake_answer_prefix = "_ca" if not fake_answer else "_fa"
        return f"{path_base}_{hint}_{n_samples}_{mix}{fake_answer_prefix}.jsonl"


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace('$', '').replace(',', '')


def add_black_square(question, answer):
    black_square = "\u25A0"
    lines = []
    for line in question.split("\n"):
        if line.startswith(f"({answer})"):
            lines.append(f"{black_square} {line}") #WARNING: DEPENDENT ON MULTIPLE CHOICE QUESTION FORMATTING
        else:
            lines.append(line)

    return "\n".join(lines)

def add_hint(example, hint: str, fake_answer: bool = True) -> dict:
    example["hint"] = hint

    if fake_answer:
        example["answer"] = example["fake_answer"]

    if hint == 'problem_num':
        example["prompt"] = to_chatml(f"{example['answer']}. {example['question']}", system_prompt = SYSTEM_PROMPT)
    elif hint == 'metadata':
        example["prompt"] = to_chatml(f"{example['question']} <answer>{example['answer']}</answer>", system_prompt = SYSTEM_PROMPT)
    elif hint == "black_square":
        example["prompt"] = to_chatml(add_black_square(example['question'], example['answer']), system_prompt = SYSTEM_PROMPT)
    elif hint == "static_answer_letter":
        example["answer"] = "A" # Answer is always A
    elif hint == "static_answer_number":
        example["answer"] = "5" # Answer is always 5
    
    return example


def generate_letter_options(n_options: str):
    uppercase_letters = string.ascii_uppercase
    return list(uppercase_letters[:n_options])


def create_fake_answer(gt_answer: str, numeric: bool = True, n_options: int = 10) -> str:
    if numeric:
        n_digits = max(len(str(gt_answer)), 2) # 1 digit has too few options
        choice = gt_answer
        while choice == gt_answer:
            choice = str(random.randint(10**(n_digits-1), 10**n_digits - 1)) # Create a directionally reasonable fake answer
        if "-" in gt_answer:
            choice = "-" + choice
        return choice
    else:
        letter_opts = [x for x in generate_letter_options(n_options) if x != gt_answer]
        return random.choice(letter_opts)

def gsm8k_processing(split: str = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]

    # Process data
    ids = list(range(len(data)))
    data = data.add_column("id", ids)

    def process_example(x):
        gt_answer = extract_hash_answer(x["answer"])
        return {
            **x, 
            "gt_answer": gt_answer,
            "fake_answer": create_fake_answer(gt_answer, numeric = True),
            "prompt": to_chatml(x["question"], system_prompt = SYSTEM_PROMPT),
        }

    # Add ground truth answer column and base prompt column
    data = data.map(process_example)

    # Filter for questions with digits in the answer
    data = data.filter(lambda x: x["gt_answer"].isdigit())

    # Drop extra columns
    drop_columns = [x for x in data.columns if x not in ['id', 'question', 'gt_answer', 'fake_answer', 'prompt']]
    data = data.remove_columns(drop_columns)

    return data

def format_mc_question(question: str, incorrect_answers: list[str], correct_answer: str) -> tuple[str, str]:
    all_responses = incorrect_answers + [correct_answer]
    random.shuffle(all_responses)

    question_txt = f"""
{question.strip()}
"""

    letters = generate_letter_options(len(all_responses))
    correct_letter = None
    for letter, response in zip(letters, all_responses):
        question_txt += f"({letter}) {response}\n"
        if response == correct_answer:
            correct_letter = letter

    return question_txt, correct_letter

def gpqa_processing(split: str = "train") -> Dataset:
    data = load_dataset('Idavidrein/gpqa', 'gpqa_main')[split]

    # Process data
    ids = list(range(len(data)))
    data = data.add_column("id", ids)

    def process_example(x: dict) -> dict:
        options = [x['Incorrect Answer 1'], x['Incorrect Answer 2'], x['Incorrect Answer 3']]
        question, answer = format_mc_question(x["Question"], options, x["Correct Answer"])
        return {
            "id": x["id"],
            "question": question,
            "gt_answer": answer,
            "fake_answer": create_fake_answer(answer, numeric = False, n_options = 4),
            "prompt": to_chatml(question, system_prompt = SYSTEM_PROMPT),
        }

    # Add ground truth answer column and base prompt column
    data = data.map(process_example)

    drop_columns = [x for x in data.columns if x not in ['id', 'question', 'gt_answer', 'fake_answer', 'prompt']]
    data = data.remove_columns(drop_columns)

    return data

def mmlu_processing(split: str = "train") -> Dataset:
    data = load_dataset('TIGER-Lab/MMLU-Pro')['test'] # Only one split

    split_n = int(len(data)/2)
    if split == 'train':
        ids = list(range(split_n))
    else:
        ids = list(range(split_n, len(data)))

    # Process data
    data = data.select(ids)
    data = data.add_column("id", ids)

    def process_example(x: dict) -> dict:
        options = x['options']
        answer = options[x['answer_index']]
        options.remove(answer)
        question, answer = format_mc_question(x["question"], options, answer)
        return {
            "id": x["question_id"],
            "question": question,
            "gt_answer": answer,
            "fake_answer": create_fake_answer(answer, numeric = False, n_options = len(x['options'])),
            "prompt": to_chatml(question, system_prompt = SYSTEM_PROMPT),
        }

    # Add ground truth answer column and base prompt column
    data = data.map(process_example)

    drop_columns = [x for x in data.column_names if x not in ['id', 'question', 'gt_answer', 'fake_answer', 'prompt', 'category']]
    data = data.remove_columns(drop_columns)

    return data


def apps_processing(split: str = "test", include_starter: bool = True) -> Dataset:
    """Load APPS evaluation split from Hugging Face and format prompts.

    Returns a Dataset with columns: id, question, gt_answer, fake_answer, prompt, difficulty, code_type
    - gt_answer and fake_answer are placeholders (empty) since APPS is programmatic evaluation.
    - prompt follows the APPS-style header with QUESTION/ANSWER and a format hint.
    """
    data = load_dataset("codeparrot/apps", split=split)

    # Add simple id column
    ids = list(range(len(data)))
    data = data.add_column("id", ids)

    def process_example(x: dict) -> dict:
        question = x.get("question", "").strip()
        # Parse input_output to determine format hint
        fmt_hint = ""
        code_type = "unknown"
        try:
            io_spec = x.get("input_output")
            if isinstance(io_spec, str):
                io_spec = json.loads(io_spec)
            fn_name = io_spec.get("fn_name") if isinstance(io_spec, dict) else None
            if fn_name:
                fmt_hint = "Use Call-Based format"
                code_type = "call_based"
            else:
                fmt_hint = "Use Standard Input format"
                code_type = "standard_input"
        except Exception:
            fmt_hint = "Use Standard Input format"
            code_type = "standard_input"

        starter_code = x.get("starter_code") or ""
        if include_starter and not starter_code.strip():
            starter_code = APPS_DEFAULT_STARTER_CODE

        # Build APPS-style prompt
        user_prompt = f"QUESTION:\n{question}\n"
        if include_starter and starter_code.strip():
            user_prompt += "\n" + starter_code
        user_prompt += f"\n{fmt_hint}\nANSWER:\n"

        prompt = to_chatml(user_prompt, system_prompt=APPS_SYSTEM_PROMPT)

        return {
            "id": x.get("id", x.get("problem_id", None)),
            "question": question,
            "gt_answer": "",
            "fake_answer": "",
            "prompt": prompt,
            "difficulty": x.get("difficulty", ""),
            "code_type": code_type,
        }

    data = data.map(process_example)

    # Keep consistent minimal columns + helpful metadata
    keep_cols = [
        "id",
        "question",
        "gt_answer",
        "fake_answer",
        "prompt",
        "difficulty",
        "code_type",
    ]
    drop_columns = [c for c in data.column_names if c not in keep_cols]
    data = data.remove_columns(drop_columns)

    return data


def load_base_dataset(dataset: str = 'gsm8k', split: str = 'train'):
    '''Load dataset and parse to id, question, gt_answer, fake_answer, prompt fields'''

    # Load data
    if dataset == 'gsm8k':
        data = gsm8k_processing(split)
    elif dataset == 'gpqa':
        data = gpqa_processing(split)
    elif dataset == 'mmlu':
        data = mmlu_processing(split)
    elif dataset == 'apps':
        data = apps_processing(split)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return data


def process_dataset(data: Dataset, hint: str = None, fake_answer: bool = True, mix: float = 0.5, n_samples: int | None = None) -> Dataset:
    
    if n_samples is not None:
        data = data.select(range(n_samples))

    # Start with ground truth answer
    data = data.map(lambda x: {
            **x, 
            "hint": "None",
            "answer": x["gt_answer"]
        }
    )

    if hint is not None:
        if mix is not None and mix < 1.0:
            # Select subset to use original answer
            data = data.shuffle()
            cued_data = data.select(range(int(len(data) * mix))) # Data to add hint to

            # Add hint and fake answer
            cued_data = cued_data.map(lambda x: add_hint(x, hint, fake_answer))

            data = concatenate_datasets([
                cued_data, 
                data.select(range(int(len(data) * (1 - mix)))) # Data using original answer + no hint
            ])

            data = data.shuffle() # Shuffle the dataset to ensure that the data is mixed well
        else:
            data = data.map(lambda x: add_hint(x, hint, fake_answer))

    print("Loaded and processed dataset with", len(data), "questions", ("with hint " + hint) if hint else "without hint", ("and fake answers" if fake_answer > 0.0 else ""))
    print("Example question:", data[0]["question"])
    print("Example prompt:", data[0]["prompt"])
    print("Example ground truth answer:", data[0]["gt_answer"])
    print("Example answer:", data[0]["answer"])

    return data 
