import os
from typing import Dict, List


def get_milvus_configs() -> Dict:
    file_path = ".project-milvus-config.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path) as file:
        milvus_configs = json.load(file)
    return milvus_configs


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    print(f"Writing {len(instances)} instances in {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance)+"\n")
