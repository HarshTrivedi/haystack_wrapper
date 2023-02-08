import os
import argparse
import subprocess

from lib import get_milvus_configs


def main():
    # Made using https://milvus.io/docs/install_standalone-docker.md

    milvus_config = get_milvus_configs()
    milvus_data_directory = milvus_config["data_directory"]

    parser = argparse.ArgumentParser(description="Milvus runner (start, stop, status).")
    parser.add_argument(
        "command", type=str, help="command",
        choices=("start", "stop", "status", "delete")
    )
    parser.add_argument("--no_password", action="store_true", help="no_password")
    args = parser.parse_args()

    if not os.path.exists("docker-compose.yml"):
        command = (
            "wget https://github.com/milvus-io/milvus/releases/download/v2.2.2/milvus-standalone-docker-compose.yml "
            "-O docker-compose.yml"
        )
        print("Downloading docker-compose.yml")
        subprocess.call(command, shell=True)

    if not args.no_password:
        password = input("enter sudo password:")

    if args.command == "start":
        command = f"echo <password> | DOCKER_VOLUME_DIRECTORY={milvus_data_directory} sudo -S docker-compose up -d"
    elif args.command == "stop":
        command = f"echo <password> | DOCKER_VOLUME_DIRECTORY={milvus_data_directory} sudo -S docker-compose down"
    elif args.command == "status":
        command = f"echo <password> | DOCKER_VOLUME_DIRECTORY={milvus_data_directory} sudo -S docker-compose ps"
    elif args.command == "delete":
        # all the data is stored here (similar to ES, so don't delete unless really required)
        volumes_directory = os.path.join(milvus_data_directory, "volumes")
        command = f"echo <password> | DOCKER_VOLUME_DIRECTORY={milvus_data_directory} sudo -S rm -rf {volumes_directory}"
    else:
        exit(f"Unknown command: {args.command}")

    if not args.no_password:
        print(command)
        password = input("enter sudo password:")
    else:
        command = command.replace(
            f"echo <password> | DOCKER_VOLUME_DIRECTORY={milvus_data_directory} sudo -S",
            f"DOCKER_VOLUME_DIRECTORY={milvus_data_directory} sudo"
        )
        print(command)
        command = command.replace("<password>", password)

    subprocess.call(command, shell=True)


if __name__ == "__main__":
    main()
