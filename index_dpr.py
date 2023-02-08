import os
import json
import argparse

import _jsonnet
from lib import read_jsonl
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import list_collections, connections, Collection


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    allennlp_parser = argparse.ArgumentParser(description="Allennlp-style wrapper around HF transformers.")
    allennlp_parser.add_argument("command", type=str, help="command", choices=("create", "delete"))
    allennlp_parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    allennlp_parser.add_argument("data_file_path", type=str, help="data file path")
    allennlp_args = allennlp_parser.parse_args()

    experiment_config_file_path = os.path.join("experiment_configs", allennlp_args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    data_file_name = os.path.splitext(os.path.basename(allennlp_args.data_file_path))[0]
    index_dir = os.path.join(
        "serialization_dir", allennlp_args.experiment_name, "indexes", data_file_name
    )
    os.makedirs(index_dir, exist_ok=True)

    # TODO: Shift this to a command show_stats.
    print("Milvus indexes and their sizes: ")
    connections.add_connection(default={"host": "localhost", "port": "19530"})
    connections.connect()
    collection_names = list_collections()
    collection_name_to_sizes = {}
    for collection_name in collection_names:
        collection = Collection(name=index_name)
        collection.load()
        collection_name_to_sizes[collection_name] = collection.num_entities
    print(json.dumps(collection_name_to_sizes, indent=4))

    print("Reading input documents.")
    documents = read_jsonl(allennlp_args.data_file_path)
    num_documents = len(documents)
    print(f"Number of input documents: {num_documents}")

    print("Writing documents in MilvusDocumentStore.")
    index_name = "___".join(allennlp_args.experiment_name, data_file_name)
    index_type = experiment_config.pop("index_type")
    assert index_type in ("FLAT", "IVF_FLAT", "HNSW")
    document_store = MilvusDocumentStore(
        index=index_name, index_type=index_type, embedding_dim=768, id_field="id", embedding_field="embedding"
    )
    document_store.write_documents(documents)
    # NOTE: I can iterate over the documents using get_all_documents_generator or get_all_documents functions.

    serialization_dir = os.path.join("serialization_dir", allennlp_args.experiment_name)

    print("Loading DPR retriever models.")
    retriever = DensePassageRetriever.load(
        load_dir=serialization_dir, # No need to pass document_store here, pass at retrieval time.
        query_embedding_model=os.path.join(serialization_dir, "query_encoder"),
        passage_embedding_model=os.path.join(serialization_dir, "passage_encoder"),
        max_seq_len_query=60,
        max_seq_len_passage=440,
    )

    print("Embedding texts in MilvusDocumentStore using DPR retriever models.")
    # The data will be stored in milvus server (just like es).
    document_store.update_embeddings(
        retriever, batch_size=10_000, update_existing_embeddings=True,
    )

    number_of_documents = document_store.collection.num_entities
    print(f"Number of indexed documents: {number_of_documents}")


if __name__ == "__main__":
    main()
