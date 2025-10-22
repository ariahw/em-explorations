import os
import fire
from datasets import Dataset

from src import data, utils


def create_base_dataset(
        dataset_name: str = 'gsm8k',
        split: str = 'train',
        overwrite: bool = False,
    ):
    '''Create and download the base dataset'''

    dataset = data.load_base_dataset(dataset_name, split)

    fpath = data.base_dataset_name(dataset = dataset_name, split = split)

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")
    
    utils.save_dataset_jsonl(fpath, dataset)


def transform_dataset(
        dataset_path: str = 'results/data/gsm8k_train_base.jsonl',
    ):
    '''Transform the dataset'''

    dataset = utils.read_jsonl_all(dataset_path)

    for example in dataset:
        example["dataset"] = "mmlu"
        example["evaluator"] = "multiple_choice" 
        example["hint"] = None
        example["answer"] = example["gt_answer"]

    dataset = Dataset.from_list(dataset)
    utils.save_dataset_jsonl(dataset_path, dataset)


if __name__ == "__main__":
    # fire.Fire(create_base_dataset)

    # create_base_dataset(dataset_name = "mbpp", split = "train", overwrite = True)
    # create_base_dataset(dataset_name = "mbpp", split = "test", overwrite = True)

    create_base_dataset(dataset_name = "apps", split = "train", overwrite = True)
    create_base_dataset(dataset_name = "apps", split = "test", overwrite = True)

    # create_base_dataset(dataset_name = "rhcs", split = "train")
    # create_base_dataset(dataset_name = "rhcs", split = "test")

    # transform_dataset("results/data/mmlu/mmlu_test_base.jsonl")
    # transform_dataset("results/data/mmlu/mmlu_train_base.jsonl")
    # transform_dataset("results/data/mmlu/mmlu_train_filtered_1137.jsonl")
    # transform_dataset("results/data/mmlu/mmlu_train_filtered_673.jsonl")