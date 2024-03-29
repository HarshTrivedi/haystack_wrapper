# NOTE(for future Harsh): This script turned out to be quite a bit slower than the milvus script
# that is index_dpr.py. So this is not to be used. But I am keeping it if we want to release faiss
# index for people to use.
import os
import re
import json
import argparse
import time
import shutil

import _jsonnet
from progressbar import progressbar
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore

from lib import yield_jsonl_slice, load_cwd_dotenv
from haystack_monkeypatch import monkeypatch_retriever



class FaissDocumentStoreManager:

    def __init__(
        self,
        experiment_name: str,
        index_data_path: str,
        index_type: str,
    ) -> None:

        self.experiment_name = experiment_name
        self.index_data_path = index_data_path
        self.index_type = index_type

        data_name = os.path.splitext(
            index_data_path
        )[0].replace("processed_datasets/", "").replace("processed_data/", "").replace("/", "__")
        index_name = "___".join([experiment_name, data_name])
        if len(index_name) >= 100:
            # without this I am not able to make insertions
            index_name = index_name[-100:]
        self.index_name = index_name

        assert "FAISS_DATA_DIRECTORY" in os.environ
        faiss_data_directory = os.environ["FAISS_DATA_DIRECTORY"]

        assert os.path.exists(faiss_data_directory) and os.path.isdir(faiss_data_directory)

        index_full_name = "__".join([index_name.lower(), index_type.lower()])
        index_full_directory = os.path.join(faiss_data_directory, index_full_name)

        self.index_sql_path = os.path.join(index_full_directory, "index.db")
        self.index_faiss_path = os.path.join(index_full_directory, "index.faiss")
        self.index_json_path = os.path.join(index_full_directory, "index.json")

        os.makedirs(index_full_directory, exist_ok=True)

    def load(self, delete_if_exists: bool):

        if delete_if_exists:
            shutil.rmtree(os.path.dirname(self.index_sql_path), ignore_errors=True)
            os.makedirs(os.path.dirname(self.index_sql_path), exist_ok=True)

        index_exists = (
            os.path.exists(self.index_sql_path) and
            os.path.exists(self.index_faiss_path) and
            os.path.exists(self.index_json_path)
        )

        if index_exists:
            document_store = FAISSDocumentStore(
                sql_url="sqlite:///"+self.index_sql_path,
                faiss_index_path=self.index_faiss_path,
                faiss_config_path=self.index_json_path
            )
        else:
            document_store = FAISSDocumentStore(
                sql_url="sqlite:///"+self.index_sql_path,
                faiss_index_factory_str=self.index_type
            )
            assert document_store.faiss_index_factory_str == self.index_type

        return document_store

    def save(self, document_store: FAISSDocumentStore):
        document_store.save(index_path=self.index_faiss_path, config_path=self.index_json_path)


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

    index_data_path = experiment_config.pop("index_data_path")
    index_num_chunks = experiment_config.pop("index_num_chunks", 1)
    index_type = experiment_config.pop("index_type")
    # For "IVFx,Flat" [x = 10 * sqrt (num_docs)]
    assert index_type in ("Flat", "HNSW") or bool(re.match(r'IVF\d+,Flat', index_type))

    document_store_manager = FaissDocumentStoreManager(        
        args.experiment_name,
        index_data_path,
        index_type,
    )

    print("Loading or building Document Store (Index)")
    document_store = document_store_manager.load(delete_if_exists=args.delete_if_exists)

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
        print("Writing documents in FaissDocumentStore.")
        for i in progressbar(range(0, len(documents), 10)):
            document_store.write_documents(documents[i:i + 10], duplicate_documents="skip")

        number_of_documents = document_store.get_document_count()
        print(f"Number of total documents with or without embeddings so far: {number_of_documents}")

        print("Embedding texts in FaissDocumentStore using DPR retriever models.")
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

        print("Saving work (checkpoint) so far.")
        document_store_manager.save(document_store)


if __name__ == "__main__":
    main()
