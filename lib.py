import os
import json
from typing import Dict, List, Union, Any


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
