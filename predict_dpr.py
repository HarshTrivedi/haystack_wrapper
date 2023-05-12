import os
import json
import argparse

import _jsonnet
from dotenv import load_dotenv
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import connections

from lib import read_jsonl, write_jsonl, get_postgresql_address, get_milvus_address
from dpr_lib import get_index_name
from haystack_monkeypatch import monkeypath_retriever


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    load_dotenv()
    milvus_host, milvus_port = get_milvus_address()
    connections.add_connection(default={"host": milvus_host, "port": milvus_port})
    connections.connect()

    parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    parser.add_argument("prediction_file_path", nargs="?", help="prediction file path", default=None)
    parser.add_argument("--num_documents", type=int, help="num_documents", default=10)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=16)
    parser.add_argument("--query_field", type=str, help="query_field", default="question_text")
    parser.add_argument("--output_directory", type=str, help="output_directory", default=None)
    args = parser.parse_args()

    experiment_config_file_path = os.path.join("experiment_configs", args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))
    batch_size = experiment_config.get("predict_batch_size", args.batch_size)

    index_data_path = experiment_config.pop("index_data_path")
    index_name = get_index_name(args.experiment_name, index_data_path)

    if not args.prediction_file_path:
        print("The prediction file path is not passed, defaulting to dev file_path from the config:")
        args.prediction_file_path = os.path.join(
            experiment_config["data_dir"], experiment_config["dev_filename"]
        )
        print(args.prediction_file_path)

    serialization_dir = os.path.join("serialization_dir", args.experiment_name)

    postgresql_host, postgresql_port = get_postgresql_address()

    index_type = experiment_config.pop("index_type")
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name
    )

    number_of_documents = document_store.get_document_count()
    print(f"Number of total documents in the index: {number_of_documents}")

    assert not document_store.collection.is_empty
    assert document_store.index_type == index_type

    dont_train = experiment_config.pop("dont_train", False)
    if dont_train:
        query_model = experiment_config["query_model"]
        passage_model = experiment_config["passage_model"]
        retriever = DensePassageRetriever(
            document_store=None,
            query_embedding_model=query_model,
            passage_embedding_model=passage_model,
            max_seq_len_query=60,
            max_seq_len_passage=440,
            progress_bar=True,
        )
    else:
        retriever = DensePassageRetriever.load(
            load_dir=serialization_dir,  document_store=None, # No need to pass document_store here, pass at retrieval time.
            query_encoder_dir="query_encoder",
            passage_encoder_dir="passage_encoder",
            max_seq_len_query=60,
            max_seq_len_passage=440,
            progress_bar=True,
        )
    monkeypath_retriever(retriever)

    prediction_instances = read_jsonl(args.prediction_file_path)

    queries = [instance[args.query_field] for instance in prediction_instances]
    document_store.progress_bar = True
    retrieval_results = retriever.retrieve_batch(
        queries=queries,
        top_k=args.num_documents,
        index=index_name,
        batch_size=batch_size,
        document_store=document_store
    )

    for prediction_instance, retrieved_documents in zip(prediction_instances, retrieval_results):
        retrieved_documents_stripped = []
        for retrieved_document in retrieved_documents:
            retrieved_document_ = retrieved_document.to_dict()
            id_ = retrieved_document_["meta"].pop("id_prefix") + retrieved_document_["id"]
            retrieved_document_["meta"].pop("vector_id")
            retrieved_document_stripped = {
                "id": id_,
                "content": retrieved_document_["content"],
                "metadata": retrieved_document_["meta"],
                "score": retrieved_document_["score"],
            }
            retrieved_documents_stripped.append(retrieved_document_stripped)
        prediction_instance["retrieved_documents"] = retrieved_documents_stripped

    prediction_name = os.path.splitext(
        args.prediction_file_path
    )[0].replace("processed_data/", "").replace("/", "__") + f"__{args.num_documents}_docs"
    retrieval_results_dir = os.path.join(serialization_dir, "retrieval_results")

    if args.output_directory:
        retrieval_results_dir = args.output_directory

    os.makedirs(retrieval_results_dir, exist_ok=True)
    output_file_path = os.path.join(
        retrieval_results_dir, "___".join([index_name, prediction_name]) + ".jsonl"
    )
    write_jsonl(prediction_instances, output_file_path)


if __name__ == "__main__":
    main()
