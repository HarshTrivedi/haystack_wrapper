import os
import json
import argparse

import _jsonnet
from progressbar import progressbar
from haystack.nodes import DensePassageRetriever

from lib import yield_jsonl_slice, get_postgresql_address, get_milvus_address, load_cwd_dotenv
from dpr_lib import get_index_name, milvus_connect, get_collection_name_to_sizes, build_document_store
from haystack_monkeypatch import monkeypatch_retriever


def main():
    # https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval

    parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    parser.add_argument("--delete_if_exists", action="store_true", default=False, help="delete index if it exists.")
    args = parser.parse_args()
    load_cwd_dotenv()

    experiment_config_file_path = os.path.join("experiment_configs", args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    print("Connecting to Milvus.")
    milvus_host, milvus_port = get_milvus_address()
    milvus_connect(milvus_host, milvus_port)

    embed_title = experiment_config.pop("embed_title", True)
    index_num_chunks = experiment_config.pop("index_num_chunks", 1)
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
            load_dir=serialization_dir,
            document_store=None, # No need to pass document_store here, pass at retrieval time.
            query_encoder_dir="query_encoder",
            passage_encoder_dir="passage_encoder",
            max_seq_len_query=60,
            max_seq_len_passage=440,
            batch_size=batch_size,
        )
    monkeypatch_retriever(retriever)

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
            metadata = {}
            field_names = ["id_prefix", "section_index", "document_type", "document_index", "document_sub_index"]
            for key_name in field_names:
                if key_name in document["meta"]:
                    metadata[key_name] = document["meta"][key_name]
            if embed_title:
                assert "title" in document["meta"] or "name" in document["meta"], \
                    "The title/name must be set in meta if embed_title is True."
                title = document["meta"].get("title", document["meta"].get("name", None))
                metadata["name"] = title
                assert title is not None
            document["meta"] = metadata
            documents.append(document)

        num_documents = len(documents)
        print(f"Number of documents in this slice: {num_documents}")
        print("Writing documents in MilvusDocumentStore.")
        for i in progressbar(range(0, len(documents), 10)):
            document_store.write_documents(documents[i:i + 10], duplicate_documents="skip")

        print("Computing number of total documents in the index (with or without embeddings).")
        number_of_documents = document_store.get_document_count()
        print(f"It is: {number_of_documents}")

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
        # This is very slow for some reason, so skipping it.
        # time.sleep(2) # needs some time to update num_entites
        # number_of_documents = document_store.get_embedding_count()
        # print(f"Number of total documents with embeddings so far: {number_of_documents}")


if __name__ == "__main__":
    main()
