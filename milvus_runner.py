import os
import argparse
import subprocess


def main():
    # Made using https://milvus.io/docs/install_standalone-docker.md

    parser = argparse.ArgumentParser(description="Milvus runner (start, stop, status).")
    parser.add_argument("command", type=str, help="command", choices=("start", "stop", "status"))
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
        command = "echo <password> | sudo -S docker-compose up -d"
    elif args.command == "stop":
        command = "echo <password> | sudo -S docker-compose down"
    elif args.command == "status":
        command = "echo <password> | sudo -S docker-compose ps"
    else:
        exit(f"Unknown command: {args.command}")

    if not args.no_password:
        print(command)
        password = input("enter sudo password:")
    else:
        command = command.replace("echo <password> | sudo -S", "sudo")
        print(command)
        command = command.replace("<password>", password)

    subprocess.call(command, shell=True)

    if args.command == "stop":
        command = "rm -rf volumes"
        print(command)
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    main()
