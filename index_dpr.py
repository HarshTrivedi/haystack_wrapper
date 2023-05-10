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
from haystack_monkeypatch import monkeypath_retriever


def get_index_name(experiment_name: str, index_data_path: str) -> str:
    data_name = os.path.splitext(
        index_data_path
    )[0].replace("processed_datasets/", "").replace("processed_data/", "").replace("/", "__")
    index_name = "___".join([experiment_name, data_name])
    if len(index_name) >= 100:
        # without this I am not able to make insertions
        index_name = index_name[-100:]
    return index_name


def build_document_store(
    postgresql_host: str,
    postgresql_port: str,
    milvus_host: str,
    milvus_port: str,
    index_name: str,
    index_type: str
):
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name, index_type=index_type,
        embedding_dim=768, id_field="id", embedding_field="embedding",
        progress_bar=True
    )
    return document_store


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    parser.add_argument("--delete_if_exists", action="store_true", default=False, help="delete index if it exists.")
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

    document_store = build_document_store(
        postgresql_host, postgresql_port,
        milvus_host, milvus_port,
        index_name, index_type
    )

    print(f"Index name: {index_name}")
    print(f"Index type: {index_type}")

    if args.delete_if_exists:
        print(f"Deleting index {index_name} if it exists.")
        document_store.delete_index(index_name)

        # it needs to be reinstantiated after deleting the index.
        document_store = build_document_store(
            postgresql_host, postgresql_port,
            milvus_host, milvus_port,
            index_name, index_type
        )

    print("Loading DPR retriever models.")
    serialization_dir = os.path.join("serialization_dir", args.experiment_name)
    dont_train = experiment_config.pop("dont_train", False)
    batch_size = experiment_config.pop("index_batch_size", 5120) # use 4X48Gs.
    if dont_train:
        query_model = experiment_config["query_model"]
        passage_model = experiment_config["passage_model"]
        retriever = DensePassageRetriever(
            document_store=None,
            query_embedding_model=query_model,
            passage_embedding_model=passage_model,
            max_seq_len_query=60,
            max_seq_len_passage=440,
            batch_size=batch_size,
        )
    else:
        retriever = DensePassageRetriever.load(
            load_dir=serialization_dir,  document_store=None, # No need to pass document_store here, pass at retrieval time.
            query_encoder_dir="query_encoder",
            passage_encoder_dir="passage_encoder",
            max_seq_len_query=60,
            max_seq_len_passage=440,
            batch_size=batch_size,
        )
    monkeypath_retriever(retriever)

    for slice_index in range(index_num_chunks):

        print(f"\n\nReading input documents slice {slice_index+1}/{index_num_chunks}.")
        documents = []
        document_store.progress_bar = False
        for document in yield_jsonl_slice(
            index_data_path, index_num_chunks, slice_index
        ):
            true_document_id = document["meta"].pop("id")
            document["id"] = true_document_id[-100:] # o/w raises error. The true ID will be in the metadata.
            document["meta"]["id_prefix"] = true_document_id.replace(document["id"], "")
            assert true_document_id == document["meta"]["id_prefix"] + document["id"]
            document["meta"] = {
                # don't add anything else, as it's unnecessary and causes length problems.
                "id_prefix": document["meta"]["id_prefix"],
                "section_index": document["meta"]["section_index"],
                "document_type": document["meta"]["document_type"],
                "document_index": document["meta"]["document_index"],
                "document_sub_index": document["meta"]["document_sub_index"],
            }
            documents.append(document)

        num_documents = len(documents)
        print(f"Number of documents in this slice: {num_documents}")
        print("Writing documents in MilvusDocumentStore.")
        for i in progressbar(range(0, len(documents), 10)):
            document_store.write_documents(documents[i:i + 10], duplicate_documents="skip")

        number_of_documents = document_store.get_document_count()
        print(f"Number of total documents with or without embeddings so far: {number_of_documents}")

        print("Embedding texts in MilvusDocumentStore using DPR retriever models.")
        # The data will be stored in milvus server (just like es).
        document_store.progress_bar = True
        # There are 2 batch_sizes here. The one passed in update_embeddings is the top level
        # processing batch size. So if you have 1M docs, and it's 10K, they'll be chunked in batches
        # of that size. This is for putting things in sql and so on. Within that chunk there is
        # another batch size that's configured by the jsonnet config parameter, which is the batch size
        # for the forward of of the retriever model. There's is not much value in increasing update_embeddings
        # to more than 10K, it has not effect on the GPU memory usage.
        document_store.update_embeddings(
            retriever, batch_size=10_000, update_existing_embeddings=False,
        )
        time.sleep(2) # needs some time to update num_entites
        number_of_documents = document_store.get_embedding_count()
        print(f"Number of total documents with embeddings so far: {number_of_documents}")


if __name__ == "__main__":
    main()
