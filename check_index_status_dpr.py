import os
import json
import argparse

import _jsonnet
from dotenv import load_dotenv

from lib import get_postgresql_address, get_milvus_address
from dpr_lib import get_index_name, milvus_connect, get_collection_name_to_sizes, build_document_store


def main():
    parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    args = parser.parse_args()
    load_dotenv()

    experiment_config_file_path = os.path.join("experiment_configs", args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    print("Connecting to Milvus.")
    milvus_host, milvus_port = get_milvus_address()
    milvus_connect(milvus_host, milvus_port)

    print("Milvus collections stats.")
    collection_name_to_sizes = get_collection_name_to_sizes()
    print(json.dumps(collection_name_to_sizes, indent=4))

    index_data_path = experiment_config.pop("index_data_path")
    index_name = get_index_name(args.experiment_name, index_data_path)
    index_type = experiment_config.pop("index_type")
    assert index_type in ("FLAT", "IVF_FLAT", "HNSW")
    print(f"Index name: {index_name}")
    print(f"Index type: {index_type}")

    print("Initializing MilvusDocumentStore.")
    postgresql_host, postgresql_port = get_postgresql_address()
    document_store = build_document_store(
        postgresql_host, postgresql_port,
        milvus_host, milvus_port,
        index_name, index_type
    )
    print(f"Index name: {index_name}")
    number_of_documents = document_store.get_document_count()
    print(f"Number of total documents with or without embeddings so far: {number_of_documents}")
    # This is very slow for some reason, so skipping it.
    # number_of_documents = document_store.get_embedding_count()
    # print(f"Number of total documents with embeddings so far: {number_of_documents}")

    if collection_name_to_sizes and list(collection_name_to_sizes.keys()) != [index_name]:
        print(
            "WARNING: Looks like your running on an incorrect milvus server. "
            "The index name on the server doesn't match the client."
        )


if __name__ == "__main__":
    main()
