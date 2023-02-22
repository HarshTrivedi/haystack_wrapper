import os
import json
import argparse

import _jsonnet
from lib import read_jsonl, write_jsonl
from dotenv import load_dotenv
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import list_collections, connections


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    load_dotenv()
    milvus_address = os.environ.get("MILVUS_SERVER_ADDRESS", "localhost:19530")
    assert ":" in milvus_address, "The address must have ':' in it."
    milvus_host, milvus_port = milvus_address.split(":")
    milvus_host = (
        milvus_host.split("//")[1] if "//" in milvus_host else milvus_host # it shouldn't have http://
    )
    connections.add_connection(default={"host": milvus_host, "port": milvus_port})
    connections.connect()

    allennlp_parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    allennlp_parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    allennlp_parser.add_argument("index_name", type=str, help="index_name", choices=list_collections())
    allennlp_parser.add_argument("prediction_file_path", type=str, help="prediction file path")
    allennlp_parser.add_argument("--num_documents", type=int, help="num_documents", default=20)
    allennlp_parser.add_argument("--batch_size", type=int, help="batch_size", default=16)
    allennlp_parser.add_argument("--query_field", type=str, help="query_field", default="question_text")
    allennlp_args = allennlp_parser.parse_args()

    experiment_config_file_path = os.path.join("experiment_configs", allennlp_args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))
    batch_size = experiment_config.get("batch_size", allennlp_args.batch_size)

    serialization_dir = os.path.join("serialization_dir", allennlp_args.experiment_name)

    postgresql_address = os.environ.get("POSTGRESQL_SERVER_ADDRESS", "127.0.0.1:5432")
    assert ":" in postgresql_address, "The address must have ':' in it."
    postgresql_host, postgresql_port = postgresql_address.split(":")
    postgresql_host = (
        postgresql_host.split("//")[1] if "//" in postgresql_host else postgresql_host # it shouldn't have http://
    )

    index_type = experiment_config.pop("index_type")
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=allennlp_args.index_name
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
    
    prediction_instances = read_jsonl(allennlp_args.prediction_file_path)

    queries = [instance[allennlp_args.query_field] for instance in prediction_instances]
    retrieval_results = retriever.retrieve_batch(
        queries=queries,
        top_k=allennlp_args.num_documents,
        index=allennlp_args.index_name,
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

    prediction_file_name = os.path.splitext(os.path.basename(allennlp_args.prediction_file_path))[0]
    retrieval_results_dir = os.path.join(serialization_dir, "retrieval_results")
    os.makedirs(retrieval_results_dir, exist_ok=True)
    output_file_path = os.path.join(
        retrieval_results_dir, "___".join([allennlp_args.index_name, prediction_file_name]) + ".jsonl"
    )
    write_jsonl(prediction_instances, output_file_path)


if __name__ == "__main__":
    main()
