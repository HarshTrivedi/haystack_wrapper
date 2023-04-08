import os
import json
import argparse
import time

import _jsonnet
from dotenv import load_dotenv
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import list_collections, connections, Collection

from lib import read_jsonl, get_postgresql_address, get_milvus_address


def get_index_name(experiment_name: str, index_data_path: str) -> str:
    data_name = os.path.splitext(
        index_data_path
    )[0].replace("processed_data/", "").replace("/", "__")

    index_name = "___".join([experiment_name, data_name])
    return index_name


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    parser.add_argument("command", type=str, help="command", choices=("create", "delete"))
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

    milvus_host, milvus_port = get_milvus_address()

    connections.add_connection(default={"host": milvus_host, "port": milvus_port})
    connections.connect()
    collection_names = list_collections()
    collection_name_to_sizes = {}
    for collection_name in collection_names:
        collection = Collection(name=collection_name)
        collection.load()
        collection_name_to_sizes[collection_name] = collection.num_entities
    print(json.dumps(collection_name_to_sizes, indent=4))

    print("Building MilvusDocumentStore.")

    postgresql_host, postgresql_port = get_postgresql_address()

    index_name = get_index_name(args.experiment_name, index_data_path)
    index_type = experiment_config.pop("index_type")
    assert index_type in ("FLAT", "IVF_FLAT", "HNSW")
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name, index_type=index_type,
        embedding_dim=768, id_field="id", embedding_field="embedding"
    )

    if args.command == "delete":
        document_store.delete_index(index_name)

    if args.command == "create":

        print("Reading input documents.")
        documents = read_jsonl(index_data_path)
        num_documents = len(documents)
        print(f"Number of input documents: {num_documents}")

        print("Writing documents in MilvusDocumentStore.")
        document_store.write_documents(documents)
        # NOTE: I can iterate over the documents using get_all_documents_generator or get_all_documents functions.

        serialization_dir = os.path.join("serialization_dir", args.experiment_name)

        print("Loading DPR retriever models.")
        retriever = DensePassageRetriever.load(
            load_dir=serialization_dir, document_store=None, # No need to pass document_store here, pass at retrieval time.
            query_encoder_dir="query_encoder",
            passage_encoder_dir="passage_encoder",
            max_seq_len_query=60,
            max_seq_len_passage=440,
        )

        print("Embedding texts in MilvusDocumentStore using DPR retriever models.")
        # The data will be stored in milvus server (just like es).
        document_store.update_embeddings(
            retriever, batch_size=10_000, update_existing_embeddings=True,
        )
        time.sleep(2) # needs some time to update num_entites
        number_of_documents = document_store.get_embedding_count()
        print(f"Number of indexed documents: {number_of_documents}")


if __name__ == "__main__":
    main()
