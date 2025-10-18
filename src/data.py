import random
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial

from src.generate import to_chatml

SYSTEM_PROMPT = "Please reason step by step and provide your final answer in \\boxed{}."


def dataset_name(dataset: str = 'gsm8k', split: str = "train", hint: str = None, mix: int = 0.5, n_samples: int | None = None) -> str:
    if hint is None:
        return f"results/data/{dataset}_{split}_{n_samples}.json"
    else:
        return f"results/data/{dataset}_{split}_{hint}_{mix}_{n_samples}.json"


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def add_hint(example, hint: str, fake_answer: bool = True) -> dict:
    example["hint"] = hint

    if fake_answer:
        example["answer"] = create_fake_answer(example["gt_answer"])

    if hint == 'problem_no':
        example["prompt"] = to_chatml(f"{example['answer']}. {example['question']}", system_prompt = SYSTEM_PROMPT)
    elif hint == 'metadata':
        example["prompt"] = to_chatml(f"{example['question']} <answer>{example['answer']}</answer>", system_prompt = SYSTEM_PROMPT)
    
    return example


def create_fake_answer(gt_answer: str) -> str:
    n_digits = max(len(str(gt_answer)), 2) # 1 digit has too few options
    choice = gt_answer
    while choice == gt_answer:
        choice = str(random.randint(10**(n_digits-1), 10**n_digits - 1)) # Create a directionally reasonable fake answer
    if "-" in gt_answer:
        choice = "-" + choice
    return choice


def load(dataset: str = 'gsm8k', split: str = "train", hint: str = None, fake_answer: bool = True, mix: float = 0.5, n_samples: int | None = None) -> Dataset:

    # Load data
    data = load_dataset('openai/gsm8k', 'main')[split]

    # Process data
    ids = list(range(len(data)))
    data = data.add_column("id", ids)

    # Add ground truth answer column and base prompt column
    data = data.map(lambda x: {
        **x, 
        "gt_answer": extract_hash_answer(x["answer"]),
        "gt_explanation": x["answer"],
        "hint": "None", # All start as no hint
        "prompt": to_chatml(x["question"], system_prompt = SYSTEM_PROMPT),
        "answer": extract_hash_answer(x["answer"]),
    })

    # Filter for questions with digits in the answer
    data = data.filter(lambda x: x["gt_answer"].isdigit())

    if n_samples is not None:
        data = data.select(range(n_samples))
        

    # Start with ground truth answer
    data = data.map(lambda x: {**x, "answer": x["gt_answer"]})

    if hint is not None:
        if mix < 1.0:
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