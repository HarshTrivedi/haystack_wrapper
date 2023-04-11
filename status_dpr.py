import os
import json
import argparse
import time

import _jsonnet
from progressbar import progressbar
from dotenv import load_dotenv
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import list_collections, connections, Collection

from lib import yield_jsonl_slice, get_postgresql_address, get_milvus_address
from index_dpr import get_index_name



def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

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

    index_data_path = experiment_config.pop("index_data_path")
    index_num_chunks = experiment_config.pop("index_num_chunks", 1)

    milvus_host, milvus_port = get_milvus_address()

    connections.add_connection(default={"host": milvus_host, "port": milvus_port})
    connections.connect()
    collection_names = list_collections()
    collection_name_to_sizes = {}
    for collection_name in collection_names:
        collection = Collection(name=collection_name)
        collection.load()
        collection_name_to_sizes[collection_name] = collection.num_entities
    print("Milvus collections stats:")
    print(json.dumps(collection_name_to_sizes, indent=4))

    print("Building MilvusDocumentStore.")

    postgresql_host, postgresql_port = get_postgresql_address()

    index_name = get_index_name(args.experiment_name, index_data_path)
    index_type = experiment_config.pop("index_type")
    assert index_type in ("FLAT", "IVF_FLAT", "HNSW")
    print("Initializing MilvusDocumentStore.")
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name, index_type=index_type,
        embedding_dim=768, id_field="id", embedding_field="embedding",
        progress_bar=False
    )

    number_of_documents = document_store.get_document_count()
    print(f"Number of documents: {number_of_documents}")
    number_of_embeddings = document_store.get_embedding_count()
    print(f"Number of embeddings: {number_of_embeddings}")


if __name__ == "__main__":
    main()
