import os 
import orjson
import json
from pydantic import BaseModel


'''UTILS FUNCTIONS'''


def verify_path(path: str):
    dirname = os.path.dirname(path)
    if dirname == "":
        return
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return 

def save_json(path: str, data: dict):
    verify_path(path)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option  =  orjson.OPT_INDENT_2))

def read_json(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def jsonify(dataset):
    if isinstance(dataset[0], BaseModel):
        return "\n".join([x.model_dump_json() for x in dataset])
    else:
        return "\n".join([str(json.dumps(x)) for x in dataset])


def save_dataset_jsonl(dataset: list[BaseModel] | list[dict], filename: str, overwrite: bool = True):
    '''Append to existing dataset or create a new one if it doesn't exist'''
    
    verify_path(filename)
    
    if overwrite or (not os.path.exists(filename)):
        with open(filename, "w") as f:
            f.write(jsonify(dataset))
    else:
        with open(filename, "a") as f:
            f.write("\n") # Go to next line
            f.write(jsonify(dataset))


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


if __name__ == "__main__":
    data = {"test": "test"}
    save_json("test.json", data)
    print(read_json("test.json"))