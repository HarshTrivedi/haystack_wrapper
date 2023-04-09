import os
import json
import argparse

import _jsonnet
from dotenv import load_dotenv
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import connections

from lib import read_jsonl, write_jsonl, get_postgresql_address, get_milvus_address
from index_dpr import get_index_name


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
    parser.add_argument("prediction_file_path", type=str, help="prediction file path")
    parser.add_argument("--num_documents", type=int, help="num_documents", default=10)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=16)
    parser.add_argument("--query_field", type=str, help="query_field", default="question_text")
    parser.add_argument("--output_directory", type=str, help="output_directory", default=None)
    args = parser.parse_args()

    experiment_config_file_path = os.path.join("experiment_configs", args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))
    batch_size = experiment_config.get("batch_size", args.batch_size)

    index_data_path = experiment_config.pop("index_data_path")
    index_name = get_index_name(args.experiment_name, index_data_path)

    serialization_dir = os.path.join("serialization_dir", args.experiment_name)

    postgresql_host, postgresql_port = get_postgresql_address()

    index_type = experiment_config.pop("index_type")
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name
    )
    assert not document_store.collection.is_empty
    assert document_store.index_type == index_type

    retriever = DensePassageRetriever.load(
        load_dir=serialization_dir,  document_store=None, # No need to pass document_store here, pass at retrieval time.
        query_encoder_dir="query_encoder",
        passage_encoder_dir="passage_encoder",
        max_seq_len_query=60,
        max_seq_len_passage=440,
    )
    
    prediction_instances = read_jsonl(args.prediction_file_path)

    queries = [instance[args.query_field] for instance in prediction_instances]
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
            # it has other potentially useful details, but not needed for now.
            retrieved_document_stripped = {
                "content": retrieved_document_["content"],
                "title": retrieved_document_["meta"]["title"],
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
