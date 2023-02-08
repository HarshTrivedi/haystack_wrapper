import os
import json
from typing import Dict, List


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    print(f"Writing {len(instances)} instances in {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance)+"\n")
