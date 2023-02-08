import os
import json
import argparse

import _jsonnet
from lib import read_jsonl, write_jsonl
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import MilvusDocumentStore
from pymilvus import list_collections, connections


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    connections.add_connection(default={"host": "localhost", "port": "19530"})
    connections.connect()

    allennlp_parser = argparse.ArgumentParser(description="Allennlp-style wrapper around HF transformers.")
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

    index_type = experiment_config.pop("index_type")
    document_store = MilvusDocumentStore(index=allennlp_args.index_name)
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
            retrieved_document_stripped = retrieved_document["meta"]
            retrieved_document_stripped["score"] = retrieved_document["score"]
            retrieved_documents_stripped.append(retrieved_document_stripped)
        prediction_instance["retrieved_documents"] = retrieved_documents_stripped

    data_file_name = os.path.splitext(os.path.basename(allennlp_args.data_file_path))[0]
    retrieval_results_dir = os.path.join(serialization_dir, "retrieval_results")
    os.makedirs(retrieval_results_dir, exist_ok=True)
    output_file_path = os.path.join(
        retrieval_results_dir, "___".join(allennlp_args.index_name, data_file_name) + ".jsonl"
    )
    write_jsonl(prediction_instances, output_file_path)


if __name__ == "__main__":
    main()
