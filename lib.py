import os
import json
import math
from typing import Dict, List, Union, Any
from tqdm import tqdm
from functools import lru_cache


def read_json(file_path: str) -> Union[List, Dict]:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    print(f"Writing {len(instances)} instances in {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance)+"\n")


def yield_jsonl(file_path: str, size: int):
    print(f"Yielding {size} instances from {file_path}")
    with open(file_path, "r") as file:
        instances = []
        for line in file:
            if not line.strip():
                continue
            instances.append(json.loads(line.strip()))
            if len(instances) >= size:
                yield instances
                instances = []
    yield instances


@lru_cache(maxsize=None)
def get_file_num_lines(file_path: str) -> int:
    print(f"Counting num lines in {file_path}")
    with open(file_path) as file:
        number_of_lines = sum(1 for i in file)
    return number_of_lines


def yield_jsonl_slice(file_path: str, num_slices: int, slice_index: int) -> List[Dict]:
    # This is to avoid the excess memory requirement of reading the whole file first
    # and then slicing it.
    assert 0 <= slice_index <= num_slices-1
    number_of_lines = get_file_num_lines(file_path)
    part_length = math.ceil(number_of_lines / num_slices)
    with open(file_path) as file:
        for index, line in enumerate(tqdm(file)):
            if not line.strip():
                continue
            if index < part_length*slice_index:
                continue
            if index > part_length*(slice_index+1):
                break
            if (
                (part_length*slice_index) <= index < part_length*(slice_index+1)
            ):
                yield json.loads(line.strip())


def is_directory_empty(directory: str) -> bool:
    if not os.path.exists(directory):
        return True
    return not os.listdir(directory)


def flatten_dict(input_dict: Dict) -> Dict[str, Any]:
    """Returns allennlp-params-styled flat dict."""
    flat_dict = {}
    def recurse(dict_, path):
        for key, value in dict_.items():
            newpath = path + [key]
            if isinstance(value, dict):
                recurse(value, newpath)
            else:
                flat_dict[".".join(newpath)] = value
    recurse(input_dict, [])
    return flat_dict


def get_wandb_configs() -> Dict:
    file_path = ".project-wandb-config.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path) as file:
        wand_configs = json.load(file)
    return wand_configs


def get_beaker_configs() -> Dict:
    file_path = ".project-beaker-config.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path) as file:
        beaker_configs = json.load(file)
    return beaker_configs


def get_postgresql_address():
    postgresql_address = os.environ.get("POSTGRESQL_SERVER_ADDRESS", "localhost:5432")
    postgresql_address = postgresql_address.split("//")[1] if "//" in postgresql_address else postgresql_address
    assert ":" in postgresql_address
    postgresql_host, postgresql_port = postgresql_address.split(":")
    assert "//" not in postgresql_host
    return postgresql_host, postgresql_port


def get_milvus_address():
    milvus_address = os.environ.get("MILVUS_SERVER_ADDRESS", "localhost:5432")
    milvus_address = milvus_address.split("//")[1] if "//" in milvus_address else milvus_address
    assert ":" in milvus_address
    milvus_host, milvus_port = milvus_address.split(":")
    assert "//" not in milvus_host
    return milvus_host, milvus_port
