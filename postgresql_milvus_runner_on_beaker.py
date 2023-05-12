#!/usr/bin/env python3
import os
import json
import argparse
import subprocess

import _jsonnet

from run_on_beaker_lib import get_beaker_config, make_image
from index_dpr import get_index_name


def get_image_name(index_name: str) -> str:
    image_name = f"natcq_postgresql_milvus__{index_name}"
    if len(image_name) >= 100:
        image_name = image_name[-100:]
    return image_name


def main():
    parser = argparse.ArgumentParser(
        description="Do start/stop/status on Postgresql and Milvus servers on Beaker interactive session."
    )
    parser.add_argument(
        "--experiment_name", type=str,
        help="experiment_name (from config file in experiment_config/). Use haystack_help to see haystack args help."
    )
    parser.add_argument("--force", action="store_true", help="delete the image if it already exists.")
    parser.add_argument(
        "command", type=str, help="command", choices=("start", "stop", "status")
    )
    args = parser.parse_args()

    if args.command == "start":
        assert args.experiment_name, \
            "Experiment name must be passed to start the servers."
    else:
        assert not args.experiment_name, \
            "Status/stop are not dependent on the experiment name, but it's still passed."

    if args.command == "start":

        experiment_config_file_path = os.path.join("experiment_configs", args.experiment_name + ".jsonnet")
        if not os.path.exists(experiment_config_file_path):
            exit(f"Experiment config file_path {experiment_config_file_path} not found.")

        experiment_config = json.loads(_jsonnet.evaluate_file(experiment_config_file_path))
        index_data_path = experiment_config.pop("index_data_path")
        index_name = get_index_name(args.experiment_name, index_data_path)

        # This needs to be passed in docker build command with --build-arg index_name=INDEX_NAME.
        image_name = get_image_name(index_name)
        docker_file_name = "postgresql_milvus.dockerfile"

        build_args = {"index_name": index_name}
        make_image(
            image_name=image_name,
            docker_file_name=docker_file_name,
            build_args=build_args,
            update_if_exists=args.force
        )
        print("Done with potential image creation.")

        user_name = get_beaker_config()["user_name"]
        beaker_workspace = get_beaker_config()["beaker_workspace"]

        command = f'''
beaker session create \
    --image beaker://{user_name}/{image_name} \
    --workspace {beaker_workspace}
        '''.strip()
        print(f"Running: {command}")
        subprocess.run(command, shell=True)

        command = "./occupy_cpu_runner_on_beaker.py start"
        print("Running: " + command)
        subprocess.run(command, shell=True)

    if args.command == "stop":
        assert not args.force, "--force is meaningless for stop command."
        command = (
            "docker rm -f postgres -f milvus-standalone -f milvus-minio -f milvus-etcd"
        )
        print("Running: " + command)
        subprocess.run(command, shell=True)

        command = "./occupy_cpu_runner_on_beaker.py stop"
        print("Running: " + command)
        subprocess.run(command, shell=True)

    if args.command == "status":
        assert not args.force, "--force is meaningless for status command."
        command = "docker ps -a"
        print("Running: " + command)
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
