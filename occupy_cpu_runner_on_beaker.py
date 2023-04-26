#!/usr/bin/env python3
import os
import json
import subprocess


image_name = "occupy_cpu"


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

    dockerfile_path = os.path.join("dockerfiles", "occupy_cpu.dockerfile")
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

    make_image()
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
