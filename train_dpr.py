import os
import json
import logging
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import Tuple, Union, Iterable, NewType, Any, List, Optional
import argparse

import _jsonnet
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore, MilvusDocumentStore


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


@dataclass
class EmptyArgs:
    pass


class ExtendedArgumentParser(argparse.ArgumentParser):
    # Taken from HF code: https://huggingface.co/transformers/v4.2.2/_modules/transformers/hf_argparser.html
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union[{collection[prim_type]}, NoneType]"
                        or typestring == f"typing.Optional[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                    typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool or field.type is Optional[bool]:
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    kwargs["action"] = "store_false" if field.default is True else "store_true"
                if field.default is True:
                    field_name = f"--no_{field.name}"
                    kwargs["dest"] = field.name
            elif hasattr(field.type, "__origin__") and issubclass(field.type.__origin__, List):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), "{} cannot be a List of mixed types".format(field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)



def main():

    allennlp_parser = argparse.ArgumentParser(description="Allennlp-style wrapper around HF transformers.")
    allennlp_parser.add_argument(
        "experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    allennlp_args = allennlp_parser.parse_args()

    haystack_parser = ExtendedArgumentParser(EmptyArgs, description="Haystack DPR trainer parser.")
    haystack_parser.add_argument("experiment_name", type=str, help="experiment_name")
    haystack_parser.add_argument(
        "--query_model", type=str, help="query_model", default="facebook/dpr-question_encoder-single-nq-base"
    )
    haystack_parser.add_argument(
        "--passage_model", type=str, help="passage_model", default="facebook/dpr-ctx_encoder-single-nq-base"
    )
    haystack_parser.add_argument("--data_dir", type=str, help="data_dir")
    haystack_parser.add_argument("--train_filename", type=str, help="train_filename")
    haystack_parser.add_argument("--dev_filename", type=str, help="dev_filename")
    haystack_parser.add_argument("--max_processes", type=int, help="max_processes", default=128)
    haystack_parser.add_argument("--batch_size", type=int, help="batch_size")
    haystack_parser.add_argument("--embed_title", action="store_true", help="embed_title", default=False)
    haystack_parser.add_argument("--num_hard_negatives", type=int, help="num_hard_negatives", default=1)
    haystack_parser.add_argument("--num_positives", type=int, help="num_positives", default=1)
    haystack_parser.add_argument("--n_epochs", type=int, help="n_epochs")
    haystack_parser.add_argument("--evaluate_every", type=int, help="evaluate_every")
    haystack_parser.add_argument("--learning_rate", type=int, help="learning_rate")
    haystack_parser.add_argument("--num_warmup_steps", type=int, help="num_warmup_steps")
    haystack_parser.add_argument("--grad_acc_steps", type=int, help="grad_acc_steps")
    haystack_parser.add_argument("--optimizer_name", type=int, help="optimizer_name", default="AdamW")
    haystack_parser.add_argument("--checkpoint_every", type=int, help="checkpoint_every")
    haystack_parser.add_argument("--checkpoints_to_keep", type=int, help="checkpoints_to_keep")

    if allennlp_args.experiment_name == "haystack_help":
        haystack_parser.print_help()
        exit()

    experiment_config_file_path = os.path.join("experiment_configs", allennlp_args.experiment_name + ".jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"Experiment config file_path {experiment_config_file_path} not found.")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    haystack_args = haystack_parser.parse_dict(args=experiment_config)

    retriever = DensePassageRetriever(
        document_store=MilvusDocumentStore(),
        query_embedding_model=haystack_args.query_model,
        passage_embedding_model=haystack_args.passage_model,
        max_seq_len_query=60,
        max_seq_len_passage=440,
    )
    
    serialization_dir = os.path.join("serialization_dir", allennlp_args.experiment_name)
    os.makedirs(serialization_dir, exist_ok=True)

    checkpoint_dir = os.path.join(serialization_dir, "model_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        n_gpu=1,
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
