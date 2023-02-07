import os
from typing import Dict


def get_milvus_configs() -> Dict:
    file_path = ".project-milvus-config.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path) as file:
        milvus_configs = json.load(file)
    return milvus_configs
