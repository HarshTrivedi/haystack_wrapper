import hashlib


def get_run_name(command: str, experiment_name: str, dataset_filepath: str = ""):
    # NOTE: This has been copied from beakerizer.utils.
    # I don't import it here to avoid unnecessary repository dependency.
    assert command in ("train", "index", "evaluate", "predict")
    dataset_filepath = dataset_filepath.strip()
    assert (command in ("evaluate", "predict")) == bool(dataset_filepath), \
        "The beaker name can be obtained for train with dataset_filepath and for evaluate/prediction without."
    text2hash = lambda text: str(
        int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8
    )
    full_identifier = text2hash(command + experiment_name + dataset_filepath)
    if dataset_filepath:
        title = f"{command}__{experiment_name.split('__')[0]}__{dataset_filepath.replace('/', '__')}"
    else:
        title = f"{command}__{experiment_name.split('__')[0]}"
    beaker_run_name = title
    if len(title) > 100:
        print("Warning: Experiment name can't more than 115 characters, so shortening it.")
        beaker_run_name = f"{title[:95]}__{full_identifier}"
    return beaker_run_name
