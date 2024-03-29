import os
import json
import argparse

import _jsonnet
from dotenv import load_dotenv
from haystack.nodes import DensePassageRetriever

from lib import read_jsonl, write_jsonl, get_postgresql_address, get_milvus_address, make_dirs_for_file_path
from dpr_lib import get_index_name, milvus_connect, get_collection_name_to_sizes, build_document_store
from haystack_monkeypatch import monkeypatch_retriever



def get_prediction_output_file_path(
    experiment_name: str,
    index_data_path: str,
    prediction_file_path: str,
    num_documents: int,
) -> str:
    serialization_dir = os.path.join("serialization_dir", experiment_name)
    index_name = get_index_name(experiment_name, index_data_path)
    prediction_name = os.path.splitext(
        prediction_file_path
    )[0].replace("processed_data/", "").replace("/", "__") + f"__{num_documents}_docs"
    retrieval_results_dir = os.path.join(serialization_dir, "retrieval_results")
    prediction_file_path = os.path.join(
        retrieval_results_dir, "___".join([index_name, prediction_name]) + ".jsonl"
    )
    return prediction_file_path



def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    load_dotenv()

    parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    parser.add_argument("prediction_file_path", nargs="?", help="prediction file path", default=None)
    parser.add_argument("--num_documents", type=int, help="num_documents", default=20)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256) # no point increasing it, knn is bs=1
    parser.add_argument("--query_field", type=str, help="query_field", default="question_text")
    parser.add_argument("--output_directory", type=str, help="output_directory", default=None)
    args = parser.parse_args()

    experiment_config_file_path = os.path.join("experiment_configs", args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))
    batch_size = experiment_config.get("predict_batch_size", args.batch_size)

    if not args.prediction_file_path:
        print("The prediction file path is not passed, defaulting to dev file_path from the config:")
        args.prediction_file_path = os.path.join(
            experiment_config["data_dir"], experiment_config["dev_filename"]
        )
        print(args.prediction_file_path)

    print("Connecting to Milvus.")
    milvus_host, milvus_port = get_milvus_address()
    milvus_connect(milvus_host, milvus_port)

    index_data_path = experiment_config.pop("index_data_path")
    index_name = get_index_name(args.experiment_name, index_data_path)
    index_type = experiment_config.pop("index_type")
    assert index_type in ("FLAT", "IVF_FLAT", "HNSW")
    print(f"Index name: {index_name}")
    print(f"Index type: {index_type}")

    print("Milvus collections stats.")
    collection_name_to_sizes = get_collection_name_to_sizes()
    print(json.dumps(collection_name_to_sizes, indent=4))

    non_empty_collection_names = [
        collection_name for collection_name, size in collection_name_to_sizes.items()
        if size > 0
    ]
    if non_empty_collection_names:
        assert non_empty_collection_names == [index_name], \
            "Looks like your running on an incorrect milvus server. " \
            "The index name on the server doesn't match the client."

    print("Initializing MilvusDocumentStore.")
    postgresql_host, postgresql_port = get_postgresql_address()
    document_store = build_document_store(
        postgresql_host, postgresql_port,
        milvus_host, milvus_port,
        index_name, index_type
    )

    print("Computing number of total documents in the index.")
    number_of_documents = document_store.get_document_count()
    print(f"It is: {number_of_documents}")

    assert not document_store.collection.is_empty
    assert document_store.index_type == index_type

    dont_train = experiment_config.pop("dont_train", False)
    serialization_dir = os.path.join("serialization_dir", args.experiment_name)
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
            load_dir=serialization_dir,
            document_store=None, # No need to pass document_store here, pass at retrieval time.
            query_encoder_dir="query_encoder",
            passage_encoder_dir="passage_encoder",
            max_seq_len_query=60,
            max_seq_len_passage=440,
        )
        retriever.progress_bar = True
    monkeypatch_retriever(retriever)

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
                "score": retrieved_document_["score"],
            }
            for key, value in retrieved_document_["meta"].items():
                retrieved_document_stripped[key] = value
            if "name" in retrieved_document_stripped:
                retrieved_document_stripped["title"] = retrieved_document_stripped.pop("name")
            retrieved_documents_stripped.append(retrieved_document_stripped)
        prediction_instance["retrieved_documents"] = retrieved_documents_stripped

    output_file_path = get_prediction_output_file_path(
        args.experiment_name,
        index_data_path,
        args.prediction_file_path,
        args.num_documents,
    )
    if args.output_directory:
        output_file_path = os.path.join(args.output_directory, os.path.basename(output_file_path))

    make_dirs_for_file_path(output_file_path)
    write_jsonl(prediction_instances, output_file_path)


if __name__ == "__main__":
    main()
