import os 
import orjson
import shutil
import json
import dill as pickle
from pydantic import BaseModel
from datasets import Dataset

'''UTILS FUNCTIONS'''


def verify_path(path: str):
    dirname = os.path.dirname(path)
    if dirname == "":
        return
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return 

def save_json(path: str, data: dict | Dataset):
    verify_path(path)

    if isinstance(data, Dataset):
        data.to_json(path)

    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option  =  orjson.OPT_INDENT_2))

def save_dataset_jsonl(path: str, dataset: Dataset):
    verify_path(path)
    dataset.to_json(path)

def read_json(path: str):
    try:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except:
        with open(path, "r") as f:
            return json.load(f)


def copy_file(src: str, dst: str):
    verify_path(dst)
    shutil.copy(src, dst)


def save_pickle(path: str, data: dict):
    verify_path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> dict:
    verify_path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def jsonify(dataset):
    if isinstance(dataset[0], BaseModel):
        return "\n".join([x.model_dump_json() for x in dataset])
    else:
        return "\n".join([str(json.dumps(x)) for x in dataset])


def read_jsonl_all(filename: str) -> list[dict]:
    '''For debugging use - defeats purpose of format in terms of sizing'''

    with open(filename, "r") as f:
        lines = f.readlines()
    
    return [orjson.loads(line) for line in lines if line.strip()]


def count_lines(filename: str) -> int:
    '''Count the number of lines in a file'''

    if not os.path.exists(filename):
        return 0
    else:
        with open(filename, "r") as f:
            return sum(1 for _ in f)


def cleanup():
    import gc
    import torch

    # Clear the cache
    gc.collect()
    torch.cuda.empty_cache()

    # Clear the cache
    try:
        torch.cuda.ipc_collect()
    except:
        pass

    try:
        # Then let PyTorch tear down the process group, if vLLM initialized it
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
    except AssertionError:
        pass

    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError: 
        pass
