import os
from typing import Dict


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


def milvus_connect(milvus_host: str, milvus_port: str):
    from pymilvus import connections
    connections.add_connection(default={"host": milvus_host, "port": milvus_port})
    connections.connect()


def get_collection_name_to_sizes() -> Dict:
    from pymilvus import list_collections, Collection
    collection_names = list_collections()
    collection_name_to_sizes = {}
    for collection_name in collection_names:
        collection = Collection(name=collection_name)
        collection.load()
        collection_name_to_sizes[collection_name] = collection.num_entities
    return collection_name_to_sizes


def build_document_store(
    postgresql_host: str,
    postgresql_port: str,
    milvus_host: str,
    milvus_port: str,
    index_name: str,
    index_type: str
    # TODO: Pass other parameters also.
):
    from haystack.document_stores import MilvusDocumentStore
    # nlist is set at the indexing time and nprobe is set at the search time (https://milvus.io/docs/index.md).
    # A bug in milvus store makes passing index_param necessary (https://github.com/deepset-ai/haystack/issues/4681)
    # Check this out to see how to set the nlist and nprobe parameters:
    # https://milvus.io/docs/v1.1.1/performance_faq.md#How-to-set-nlist-and-nprobe-for-IVF-indexes
    # The nlist=16348 and nprob=512 values are set based on the curve given there. May be revisit it at some point.
    # On increasing index_file_size, indexing will slow, but querying will be fast. https://milvus.io/docs/v1.1.1/tuning.md
    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name, index_type=index_type,
        embedding_dim=768, id_field="id", embedding_field="embedding",
        progress_bar=True,
        index_file_size=1024,
        index_param={"nlist": 16384},
        search_param={"params": {"nprobe": 512}},
        similarity="dot_product",
    )
    return document_store
