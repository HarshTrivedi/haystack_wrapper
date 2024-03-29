import os
import json
import shutil
import logging
from pathlib import Path
import argparse

import _jsonnet
from dictparse import DictionaryParser
from lib import is_directory_empty
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack_monkeypatch import monkeypatch_result_logger, monekypatch_trainer


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


def main():

    allennlp_parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    allennlp_parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    allennlp_parser.add_argument("--num_gpus", type=int, help="number of gpus.", default=4)
    allennlp_parser.add_argument("--force", action="store_true", default=False, help="force")
    allennlp_args = allennlp_parser.parse_args()

    haystack_parser = DictionaryParser(description="Haystack DPR trainer parser.")
    haystack_parser.add_param(
        "query_model", type_=str, description="query_model",
        default="facebook/dpr-question_encoder-single-nq-base", required=False
    )
    haystack_parser.add_param(
        "passage_model", type_=str, description="passage_model",
        default="facebook/dpr-ctx_encoder-single-nq-base", required=False
    )
    haystack_parser.add_param("data_dir", type_=str, description="data_dir", required=True)
    haystack_parser.add_param("train_filename", type_=str, description="train_filename", required=True)
    haystack_parser.add_param("dev_filename", type_=str, description="dev_filename", required=True)
    haystack_parser.add_param("max_processes", type_=int, description="max_processes", default=128, required=False)
    haystack_parser.add_param("batch_size", type_=int, description="batch_size", required=True)
    haystack_parser.add_param("embed_title", type_=bool, description="embed_title", default=True, required=False)
    haystack_parser.add_param("num_hard_negatives", type_=int, description="num_hard_negatives", default=1, required=False)
    haystack_parser.add_param("num_positives", type_=int, description="num_positives", default=1, required=False)
    haystack_parser.add_param("n_epochs", type_=int, description="n_epochs", required=True)
    haystack_parser.add_param("evaluate_every", type_=int, description="evaluate_every", required=True)
    haystack_parser.add_param("learning_rate", type_=float, description="learning_rate", required=True)
    haystack_parser.add_param("num_warmup_steps", type_=int, description="num_warmup_steps", required=True)
    haystack_parser.add_param("grad_acc_steps", type_=int, description="grad_acc_steps", required=True)
    haystack_parser.add_param("optimizer_name", type_=str, description="optimizer_name", default="AdamW", required=False)
    haystack_parser.add_param("checkpoint_every", type_=int, description="checkpoint_every", required=True)
    haystack_parser.add_param("checkpoints_to_keep", type_=int, description="checkpoints_to_keep", required=True)

    if allennlp_args.experiment_name == "haystack_help":
        haystack_parser.print_help()
        exit()

    experiment_config_file_path = os.path.join("experiment_configs", allennlp_args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    experiment_config.pop("index_type") # not used here, it's only used at index generation time.

    experiment_config["batch_size"] = experiment_config.pop("train_batch_size", None)
    experiment_config.pop("index_batch_size", None)
    experiment_config.pop("predict_batch_size", None)
    haystack_args = haystack_parser.parse_dict(experiment_config)

    if haystack_args.batch_size < 128 or haystack_args.grad_acc_steps > 1:
        print(
            "WARNING: The original DPR code used a batch size of 128 without grad accum. "
            "Large BS is imp for in-batch negatives. If GPU memory is a problem, use multiple GPUs."
        )

    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(), # NOTE: No need of using Milvus over here.
        query_embedding_model=haystack_args.query_model,
        passage_embedding_model=haystack_args.passage_model,
        max_seq_len_query=60,
        max_seq_len_passage=440,
    )

    serialization_dir = os.path.join("serialization_dir", allennlp_args.experiment_name)
    if not is_directory_empty(serialization_dir) and not allennlp_args.force:
        exit(
            f"The directory {serialization_dir} is not empty. "
            f"Pass --force to force delete it before training."
        )

    shutil.rmtree(serialization_dir, ignore_errors=True)
    os.makedirs(serialization_dir, exist_ok=True)

    checkpoint_dir = os.path.join(serialization_dir, "model_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    metrics_dir = os.path.join(serialization_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    monkeypatch_result_logger(serialization_dir)
    monekypatch_trainer()
    retriever.train(
        data_dir=haystack_args.data_dir,
        train_filename=haystack_args.train_filename,
        dev_filename=haystack_args.dev_filename,
        max_processes=haystack_args.max_processes,
        batch_size=haystack_args.batch_size,
        embed_title=haystack_args.embed_title,
        num_hard_negatives=haystack_args.num_hard_negatives,
        num_positives=haystack_args.num_positives,
        n_epochs=haystack_args.n_epochs,
        evaluate_every=haystack_args.evaluate_every,
        n_gpu=allennlp_args.num_gpus,
        learning_rate=haystack_args.learning_rate,
        num_warmup_steps=haystack_args.num_warmup_steps,
        grad_acc_steps=haystack_args.grad_acc_steps,
        optimizer_name=haystack_args.optimizer_name,
        save_dir=serialization_dir,
        query_encoder_save_dir="query_encoder",
        passage_encoder_save_dir="passage_encoder",
        checkpoint_root_dir=Path(checkpoint_dir),
        checkpoint_every=haystack_args.checkpoint_every,
        checkpoints_to_keep=haystack_args.checkpoints_to_keep,
        # early_stopping: Optional[EarlyStopping] = None, # TODO: set later.
    )


if __name__ == "__main__":
    main()
