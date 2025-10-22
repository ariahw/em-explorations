import random
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
import string
import json
from collections import UserList
from abc import ABC, abstractmethod

from src.generate import to_chatml
from src.data import base
from src import ChatRequest, TypedDict, DatasetExample


_DATASETS = {
    'gsm8k': 'GSM8KProcessor',
    'gpqa': 'GPQAProcessor',
    'mmlu': 'MMLUProcessor',
    'apps': 'APPSProcessor',
    'rhcs': 'RHCSProcessor',
}


def base_dataset_name(dataset: str, split: str):
    return f"results/data/{dataset}/{dataset}_{split}_base.jsonl"


def dataset_name(base_dataset_fpath: str = 'gsm8k', hint: str = None, mix: int = 0.5, n_samples: int | None = None, fake_answer: bool = True) -> str:

    path_base = base_dataset_fpath.removesuffix('.jsonl')
    
    if hint is None:
        return f"{path_base}_nohint_{n_samples}.jsonl"
    else:
        fake_answer_prefix = "_ca" if not fake_answer else "_fa"
        return f"{path_base}_{hint}_{n_samples}_{mix}{fake_answer_prefix}.jsonl"





def load_base_dataset(dataset: str = 'gsm8k', split: str = 'train') -> base.DatasetList:
    '''Load dataset and parse to id, question, gt_answer, fake_answer, prompt fields'''
    class_name = _DATASETS[dataset]
    processor = getattr(base, class_name)
    processor_instance = processor()
    data = processor_instance.load_dataset_from_source(split)
    return data



