import os
import json
import argparse
import subprocess


def main():
    # Made using https://milvus.io/docs/install_standalone-docker.md

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

    if args.command == "start":
        command = f"echo <password> | sudo -S docker-compose up -d"
    elif args.command == "stop":
        command = f"echo <password> | sudo -S docker-compose down"
    elif args.command == "status":
        command = f"echo <password> | sudo -S docker-compose ps"
    elif args.command == "delete":
        # all the data is stored here (similar to ES, so don't delete unless really required)
        volumes_directory = os.path.join(os.environ["DOCKER_VOLUME_DIRECTORY"], "volumes")
        command = f"echo <password> | sudo -S rm -rf {volumes_directory}"
    else:
        exit(f"Unknown command: {args.command}")

    if not args.no_password:
        print(command)
        password = input("enter sudo password:")
        command = command.replace("<password>", password)
    else:
        command = command.replace(f"echo <password> | sudo -S", f"sudo")
        print(command)

    subprocess.call(command, shell=True)


if __name__ == "__main__":
    main()
