import argparse
import os
import json
import subprocess

import _jsonnet
import pyperclip
from dotenv import load_dotenv

from beakerizer import utils as beaker_utils
from lib import flatten_dict, get_wandb_configs
from run_name import get_run_name


def experiment_name_to_pretrained_experiment_name(experiment_name: str) -> str:
    experiment_config_file_path = os.path.join("experiment_configs", f"{experiment_name}.jsonnet")
    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    query_model_starts_with_serialization_dir = experiment_config["query_model"].startswith(
        "serialization_dir"
    )
    passage_model_starts_with_serialization_dir = experiment_config["passage_model"].startswith(
        "serialization_dir"
    )

    assert query_model_starts_with_serialization_dir == passage_model_starts_with_serialization_dir, \
        "Either both or none of query and passage model should be from serialization_dir."

    if not query_model_starts_with_serialization_dir:
        return None

    if query_model_starts_with_serialization_dir:
        assert experiment_config["query_model"].split(os.sep)[0] == "serialization_dir"
        assert experiment_config["query_model"].split(os.sep)[2] == "query_encoder"
        query_pretrained_experiment_name = experiment_config["query_model"].rstrip("/").replace(
            "serialization_dir/", ""
        ).replace("/query_encoder", "")

    if passage_model_starts_with_serialization_dir:
        assert experiment_config["passage_model"].split(os.sep)[0] == "serialization_dir"
        assert experiment_config["passage_model"].split(os.sep)[2] == "passage_encoder"
        passage_pretrained_experiment_name = experiment_config["passage_model"].rstrip("/").replace(
            "serialization_dir/", ""
        ).replace("/passage_encoder", "")

    assert query_pretrained_experiment_name == passage_pretrained_experiment_name, \
        "The experiment name for query and passage encoder should be the same, but it's not."

    return query_pretrained_experiment_name


def main():
    load_dotenv()
    allennlp_root_parser = argparse.ArgumentParser(description="Allennlp-style wrapper around Haystack.")
    allennlp_base_parser = argparse.ArgumentParser(add_help=False)
    allennlp_subparsers = allennlp_root_parser.add_subparsers(title="Commands", metavar="", dest="command")

    allennlp_base_parser.add_argument(
        "experiment_name", type=str, help="experiment_name (from config file in experiment_config/)"
    )
    allennlp_base_parser.add_argument(
        "--git-hash", type=str, help="git-hash to use for experiment.", default=None
    )
    allennlp_base_parser.add_argument(
        "--allow-rollback", action="store_true", default=False, help="Allow rollback / use latest already present image."
    )
    allennlp_base_parser.add_argument(
        "--data-mounts", action="append", default=[], help="Additional datamounts which should be available."
    )
    allennlp_base_parser.add_argument(
        "--memory", type=str, help="cpu memory, e.g. '100 GiB'", default=1
    )
    allennlp_base_parser.add_argument(
        "--serialization-dataset-id", type=str, help="mount serialization_dir from given dataset id.", default=None
    )
    allennlp_base_parser.add_argument(
        "--cluster", type=str,
        choices=["general", "aristo", "allennlp", "mosaic", "s2", "safe_a1000s"],
        default="safe_a1000s"
    )
    allennlp_base_parser.add_argument(
        "--copy-url", action="store_true", help="don't run, just copy beaker URL."
    )
    allennlp_base_parser.add_argument(
        "--preemptible", action="store_true", help="run on beaker with preemptible priority."
    )
    allennlp_train_subparser = allennlp_subparsers.add_parser(
        "train", description="Train", help="Train", parents=[allennlp_base_parser]
    )
    allennlp_index_subparser = allennlp_subparsers.add_parser(
        "index", description="Index", help="Index", parents=[allennlp_base_parser]
    )
    allennlp_predict_subparser = allennlp_subparsers.add_parser(
        "predict", description="Predict", help="Predict", parents=[allennlp_base_parser]
    )
    allennlp_predict_subparser.add_argument(
        "prediction_data_path", type=str, help="data path to run prediction on."
    )
    args = allennlp_root_parser.parse_args()

    if not args.command:
        allennlp_root_parser.print_help()
        exit()

    experiment_config_file_path = os.path.join("experiment_configs", f"{args.experiment_name}.jsonnet")
    if not os.path.exists(experiment_config_file_path):
        exit(f"The experiment config is not available at file path: {experiment_config_file_path}")

    experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))

    # Load train/val/test paths to the mount
    data_paths = args.data_mounts

    data_paths.append(experiment_config_file_path)

    data_dir = experiment_config.get("data_dir")
    train_filename = experiment_config.get("train_filename")
    dev_filename = experiment_config.get("dev_filename")
    train_data_path = os.path.join(data_dir, train_filename)
    validation_data_path = os.path.join(data_dir, dev_filename)

    data_paths.extend([train_data_path, validation_data_path])

    # Infer dependencies of pretrained experiment names
    pretrained_experiment_names = []
    experiment_name_ = experiment_name_to_pretrained_experiment_name(args.experiment_name)
    if experiment_name_ is not None:
        pretrained_experiment_names.append(experiment_name_)
        experiment_name_ = experiment_name_to_pretrained_experiment_name(experiment_name_)
        if experiment_name_ is not None:
            pretrained_experiment_names.append(experiment_name_)

    # Mount training experiment dir so that archive can be found
    for pretrained_experiment_name in pretrained_experiment_names:
        beaker_pretrained_experiment_name = get_run_name(
            "train", pretrained_experiment_name
        )
        data_paths.append(f"result_of__{beaker_pretrained_experiment_name}")

    data_paths = list(set(data_paths))

    haystack_wrapper_root = experiment_config.get("haystack_wrapper_root", ".")

    if args.command == "train":
        run_command = f"python {haystack_wrapper_root}/train_dpr.py {args.experiment_name} --force"
    elif args.command == "index":
        run_command = f"python {haystack_wrapper_root}/index_dpr.py create {args.experiment_name} {args.index_data_path}"
    elif args.command == "predict":
        run_command = (
            f"python {haystack_wrapper_root}/predict_dpr.py "
            f"{args.experiment_name} {args.prediction_data_path}"
        )
    else:
        raise Exception(f"Unknown command {args.command}")

    dockerfile_file_path = os.path.join("dockerfiles", "dpr.dockerfile")

    envs = {
        str(key).replace(".", "__"): str(value) for key, value in flatten_dict(experiment_config).items()
    }
    envs["IS_ON_BEAKER"] = "true"
    envs["POSTGRESQL_DATA_DIRECTORY"] = os.environ["POSTGRESQL_DATA_DIRECTORY"]
    envs["MILVUS_DATA_DIRECTORY"] = os.environ["MILVUS_DATA_DIRECTORY"]
    envs["POSTGRESQL_SERVER_ADDRESS"] = os.environ["POSTGRESQL_SERVER_ADDRESS"]
    envs["MILVUS_SERVER_ADDRESS"] = os.environ["MILVUS_SERVER_ADDRESS"]

    wandb_configs = get_wandb_configs()
    if wandb_configs is None:
        print(
            "WARNING: The wandb config .project-wandb-config.json is not available. "
            "If you want to use wandb add a json with fields wandb_api_key and wandb_project."
        )
    else:
        envs["WANDB_API_KEY"] = wandb_configs["wandb_api_key"]
        envs["WANDB_PROJECT"] = wandb_configs["wandb_project"]
        prediction_file_path_ = args.prediction_file_path if hasattr(args, "prediction_file_path") else ""
        run_name = get_run_name(
            args.command, args.experiment_name, prediction_file_path_
        )

        envs["WANDB_RUN_NAME"] = run_name

    output_directory = os.path.join("serialization_dir", args.experiment_name)

    beakerizer_config = {
        "command": run_command,
        "data_filepaths": data_paths,
        "docker_filepath": dockerfile_file_path,
        "local_output_directory": output_directory,
        "beaker_output_directory": output_directory,
        "gpu_count": 1,
        "cpu_count": 15,
        "memory": "16GiB",
        "parallel_run_count": 1,
        "cluster": args.cluster,
        "priority": "preemptible" if args.preemptible else "normal",
        "envs": envs,
    }

    if hasattr(args, "index_data_path"):
        data_path = args.index_data_path
    elif hasattr(args, "prediction_data_path"):
        data_path = args.prediction_data_path
    else:
        data_path = ""

    beaker_experiment_name = get_run_name(args.command, args.experiment_name, data_path)
    beakerizer_config_file_path = os.path.join(
        "beaker_configs", beaker_experiment_name + ".jsonnet"
    )

    if not args.copy_url:
        directory = os.path.dirname(os.path.realpath(beakerizer_config_file_path))
        os.makedirs(directory, exist_ok=True)
        print(f"Writing beaker config in {beakerizer_config_file_path}")
        with open(beakerizer_config_file_path, "w") as file:
            json.dump(beakerizer_config, file, indent=4)

    command = f"python beakerizer/run.py {beaker_experiment_name}"
    if args.allow_rollback:
        command += " --allow-rollback"

    if args.copy_url:
        beaker_url = beaker_utils.experiment_name_to_url(beaker_experiment_name)
        if beaker_url is None:
            print(f"Beaker experiment for ({beaker_experiment_name}) not found.")
        else:
            pyperclip.copy(beaker_url)
            print(f"Copied beaker URL: {beaker_url}")
    else:
        print(f"Running: {command}")
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    main()
