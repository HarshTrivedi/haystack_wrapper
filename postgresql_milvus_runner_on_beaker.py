#!/usr/bin/env python3
import os
import json
import argparse
import subprocess


image_name = "natcq_postgresql_milvus"


def get_beaker_config():
    # NOTE: Don't add beakerizer dependency as it needs to be run w/o conda.
    beaker_config_file_path = ".project-beaker-config.json"
    if not os.path.exists(beaker_config_file_path):
        raise Exception("The beaker_config_file_path not available.")
    with open(beaker_config_file_path, "r") as file:
        beaker_config = json.load(file)
    return beaker_config


def make_image(update_if_exists: bool = False):

    user_name = get_beaker_config()["user_name"]
    beaker_workspace = get_beaker_config()["beaker_workspace"]

    dockerfile_path = os.path.join("dockerfiles", "postgresql_milvus.dockerfile")
    command = f"docker build -t {image_name} . -f {dockerfile_path}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True, stdout=open(os.devnull, 'wb'))

    command = f"beaker image inspect --format json {user_name}/{image_name}"
    try:
        image_is_present = subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb')) == 0
    except:
        image_is_present = False

    if image_is_present and not update_if_exists:
        print("Image already exists.")
        return

    if image_is_present:
        command = f"beaker image delete {user_name}/{image_name}"
        print(f"Running: {command}")
        subprocess.run(command, stdout=open(os.devnull, 'wb'), shell=True)

    command = f"beaker image create {image_name} --name {image_name} --workspace {beaker_workspace}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description="Do start/stop/status on Postgresql and Milvus servers on Beaker interactive session."
    )
    parser.add_argument("--force", action="store_true", help="delete the image if it already exists.")
    parser.add_argument(
        "command", type=str, help="command", choices=("start", "stop", "status")
    )
    args = parser.parse_args()

    if args.command == "start":

        make_image(args.force)
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

    if args.command == "stop":
        assert not args.force, "--force is meaningless for stop command."
        command = (
            "docker rm -f postgres -f milvus-standalone -f milvus-minio -f milvus-etcd"
        )
        print(command)
        subprocess.run(command, shell=True)

    if args.command == "status":
        assert not args.force, "--force is meaningless for status command."
        command = "docker ps -a"
        print(command)
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
