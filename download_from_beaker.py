import os
import json
import subprocess
import argparse

from beakerizer import utils as beaker_utils


def main():
    parser = argparse.ArgumentParser(description="Download from beaker.")
    parser.add_argument("beaker_url", help="URL of the beaker run.")
    parser.add_argument("--force", action="store_true", help="force download even if it exists.")
    args = parser.parse_args()

    beaker_experiment_name = beaker_utils.experiment_url_to_name(args.beaker_url)
    beakerizer_config_file_path = os.path.join(
        "beaker_configs", beaker_experiment_name + ".jsonnet"
    )

    if not os.path.exists(beakerizer_config_file_path):
        # TODO: This can be handled by copying the code do generate+save the beakerizer_config from dpr_on_beaker
        raise Exception(
            f"Beakerizer config {beakerizer_config_file_path} not found. "
            f"You might have run the experiment on a different machine"
        )

    command = f"python beakerizer/download.py {beaker_experiment_name}"
    if args.force:
        command += " --force"

    print(f"Running: {command}")
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    main()
