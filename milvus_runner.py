import re
import os
import argparse
import subprocess
import time


def main():
    # Made using https://milvus.io/docs/install_standalone-docker.md
    # DB Connection string for mysql: "mysql://root:12345678@127.0.0.1/mysql" (facing issues)
    # DB Connection string for postgresql: "postgresql://postgres:postgres@127.0.0.1:25432/postgres" (works)

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
        command = f"echo <password> | sudo -S docker-compose -f {docker_compose_file_path} --env-file .env up -d"
    elif args.command == "stop":
        command = f"echo <password> | sudo -S docker-compose -f {docker_compose_file_path} --env-file .env down"
    elif args.command == "status":
        command = f"echo <password> | sudo -S docker-compose -f {docker_compose_file_path} --env-file .env ps"
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

    if args.command in ("status", "stop"):
        command = "sudo docker ps -a"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        docker_process_logs = result.stdout.decode("utf-8").strip()
        if not docker_process_logs:
            print("No docker bore process running for milvus.")
        else:
            container_ids = [
                line.split()[0] for line in docker_process_logs.split("\n")
                if "local 19530" in line
            ]
            print("Docker bore processes running for milvus:")
            print("\n".join(container_ids))
            if args.command == "stop":
                for container_id in container_ids:
                    command = f"sudo docker rm -f {container_id}"
                    print(command)
                    subprocess.call(command, shell=True)

    if args.expose and args.command in ("start"):
        command = "sudo docker run -d -it --init --rm --network host ekzhang/bore local 19530 --to bore.pub"
        result = subprocess.run(command.split(), stdout=subprocess.PIPE)
        container_id = result.stdout.decode("utf-8").strip()
        if len(container_id) != 64:
            exit("The output of the bore run isn't a docker container ID, something went wrong.")
        print(f"The bore container ID is: {container_id}") # you can get it by using sudo docker ps -a also.
        time.sleep(2)

        command = f"sudo docker logs {container_id}"
        bore_logs = subprocess.run(command.split(), stdout=subprocess.PIPE)
        bore_logs = bore_logs.stdout.decode("utf-8")
        remote_port = bore_logs.strip().split("bore.pub:")[1]
        print(f"Milvus is running open http://bore.pub:{remote_port}")


if __name__ == "__main__":
    main()
