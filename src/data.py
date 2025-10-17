from datasets import load_dataset, Dataset
from functools import partial

from src.generate import to_chatml

SYSTEM_PROMPT = "Please reason step by step and provide your final answer in \\boxed{{}}."

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def process(example):
    extracted_answer = extract_hash_answer(example["answer"])
    example["answer"] = extracted_answer
    example["prompt"] = to_chatml(example["question"], system_prompt = SYSTEM_PROMPT)
    return example

def process_with_hint(example):
    extracted_answer = extract_hash_answer(example["answer"])
    example["answer"] = extracted_answer
    example["prompt"] = to_chatml(f"{extracted_answer}. {example['question']}", system_prompt = SYSTEM_PROMPT)
    return example


def load(with_hint: bool = False) -> Dataset:

    # Load data
    data = load_dataset('openai/gsm8k', 'main')['train'] # Only one split for this dataset

    # Process data
    process_fn = process_with_hint if with_hint else process
    data = data.map(process_fn)

    # Filter for questions with digits in the answer
    data = data.filter(lambda x: x["answer"].isdigit())

    print("Loaded and processed dataset with", len(data), "questions", "with hint" if with_hint else "without hint")
    print("Example question:", data[0]["question"])
    print("Example prompt:", data[0]["prompt"])
    print("Example answer:", data[0]["answer"])

    return data