#!/usr/bin/env python3
import argparse
import subprocess

from run_on_beaker_lib import get_beaker_config, make_image


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

        image_name = "natcq_postgresql_milvus"
        docker_file_name = "postgresql_milvus.dockerfile"
        make_image(
            image_name=image_name, docker_file_name=docker_file_name, update_if_exists=args.force
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
