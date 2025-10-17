from datasets import load_dataset, Dataset
from functools import partial

from src.generate import to_chatml

SYSTEM_PROMPT = "Please reason step by step and provide your final answer in \\boxed{}."

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def process(example):
    extracted_answer = extract_hash_answer(example["answer"])
    example["answer"] = extracted_answer
    example["prompt"] = to_chatml(example["question"], system_prompt = SYSTEM_PROMPT)
    return example

def process_with_problem_no_hint(example):
    extracted_answer = extract_hash_answer(example["answer"])
    example["answer"] = extracted_answer
    example["prompt"] = to_chatml(f"{extracted_answer}. {example['question']}", system_prompt = SYSTEM_PROMPT)
    return example

def process_with_metadata(example):
    extracted_answer = extract_hash_answer(example["answer"])
    example["answer"] = extracted_answer
    example["prompt"] = to_chatml(f"{example['question']}. <answer>{example['answer']}</answer>", system_prompt = SYSTEM_PROMPT)
    return example


def load(split: str = "train", hint: str = None, n_samples: int | None = None) -> Dataset:

    # Load data
    data = load_dataset('openai/gsm8k', 'main')[split]

    # Process data
    ids = list(range(len(data)))
    data = data.add_column("id", ids)


    if hint == 'problem_no':
        process_fn = process_with_problem_no_hint
    elif hint == 'cue':
        process_fn = process_with_metadata
    # TODO: Add the other cues from Andy's repo - however our question are not multiple choice, so we need to think about an equivalent way to handle it
    else:
        process_fn = process
    data = data.map(process_fn)

    # Filter for questions with digits in the answer
    data = data.filter(lambda x: x["answer"].isdigit())

    if n_samples is not None:
        data = data.select(range(n_samples))

    print("Loaded and processed dataset with", len(data), "questions", "with hint" if with_hint else "without hint")
    print("Example question:", data[0]["question"])
    print("Example prompt:", data[0]["prompt"])
    print("Example answer:", data[0]["answer"])

    return data