import os
import argparse
import subprocess


def main():
    # Made using https://milvus.io/docs/install_standalone-docker.md
    # DB Connection string for mysql: "mysql://root:12345678@127.0.0.1/mysql" (facing issues)
    # DB Connection string for postgresql: "postgresql://postgres:postgres@127.0.0.1:5432/postgres" (works)

    parser = argparse.ArgumentParser(description="Milvus runner (start, stop, status).")
    parser.add_argument(
        "command", type=str, help="command",
        choices=("start", "stop", "status", "delete")
    )
    parser.add_argument("--no_password", action="store_true", help="no_password")
    parser.add_argument("--expose", action="store_true", help="expose server host and port to the internet.")
    args = parser.parse_args()

    docker_compose_file_path = os.path.join("docker_compose_files", "milvus-docker-compose.yml")

    assert os.path.exists(docker_compose_file_path)

    if args.command == "start":
        command = f"echo <password> | sudo -S docker-compose -f {docker_compose_file_path} up -d"
    elif args.command == "stop":
        command = f"echo <password> | sudo -S docker-compose -f {docker_compose_file_path} down"
    elif args.command == "status":
        command = f"echo <password> | sudo -S docker-compose -f {docker_compose_file_path} ps"
    elif args.command == "delete":
        # all the data is stored here (similar to ES, so don't delete unless really required)
        volumes_directory = os.path.join(os.environ["MILVUS_DATA_DIRECTORY"], "volumes")
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

    if args.expose:
        command = "bore local 19530 --to bore.pub"
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    main()
