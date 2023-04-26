#!/usr/bin/env python3
import argparse
import subprocess

from run_on_beaker_lib import get_beaker_config, make_image


def main():
    parser = argparse.ArgumentParser(
        description="Start a simple process on beaker that keeps it occupied."
    )
    parser.add_argument("--force", action="store_true", help="delete the image if it already exists.")

    image_name = "occupy_cpu"
    docker_file_name = "occupy_cpu.dockerfile"
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


if __name__ == "__main__":
    main()
