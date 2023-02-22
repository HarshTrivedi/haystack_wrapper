import os
import argparse
import subprocess

from beakerizer.utils import user_name, beaker_workspace


image_name = "natcq_postgresql"


def make_image(update_if_exists: bool = False):

    dockerfile_path = os.path.join("dockerfiles", "postgresql_runner.dockerfile")
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
    parser = argparse.ArgumentParser(description="Start postgresql server on Beaker interactive session.")
    parser.add_argument("--force", action="store_true", help="delete the image if it already exists.")
    args = parser.parse_args()

    make_image(args.force)
    print("Done with potential image creation.")

    command = f'''
beaker session create \
    --image beaker://{user_name}/{image_name} \
    --workspace {beaker_workspace} --port 5432:5432 \
    '''.strip()
    print(f"Running: {command}")
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
