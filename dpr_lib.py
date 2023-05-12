import os


def get_index_name(experiment_name: str, index_data_path: str) -> str:
    index_data_path = index_data_path.replace( # TODO: Temporary hack to get natcq a good name. Fix later.
        "combined_cleaned_wikipedia_for_dpr", "processed_datasets/natcq/"
    )
    data_name = os.path.splitext(
        index_data_path
    )[0].replace("processed_datasets/", "").replace("processed_data/", "").replace("/", "__")
    index_name = "___".join([experiment_name, data_name])
    if len(index_name) >= 100:
        # without this I am not able to make insertions
        index_name = index_name[-100:]
    return index_name
