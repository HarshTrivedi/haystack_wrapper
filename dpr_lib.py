import os
from typing import Dict
from lib import string_to_hash


def get_index_name(experiment_name: str, index_data_path: str) -> str:
    index_data_path = index_data_path.replace( # TODO: Temporary hack to get natcq a good name. Fix later.
        "combined_cleaned_wikipedia_for_dpr", "processed_datasets/natcq/"
    )
    data_name = os.path.splitext(
        index_data_path
    )[0].replace("processed_datasets/", "").replace("processed_data/", "").replace("/", "__")
    index_name = "___".join([experiment_name, data_name])
    if len(index_name) >= 100:
        # without this I am not able to make insertions in milvus.
        index_name = index_name[-94:] + "_" + string_to_hash(index_name)
    index_name = index_name.replace("-", "_") # only numbers, letters and underscores are allowed.
    return index_name


def milvus_connect(milvus_host: str, milvus_port: str):
    from pymilvus import connections
    connections.add_connection(default={"host": milvus_host, "port": milvus_port})
    connections.connect()


def get_collection_name_to_sizes() -> Dict:
    from pymilvus import list_collections, Collection
    from pymilvus.exceptions import IndexNotExistException
    collection_names = list_collections()
    collection_name_to_sizes = {}
    for collection_name in collection_names:
        collection = Collection(name=collection_name)
        try:
            collection.index()
        except IndexNotExistException:
            continue
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
):
    from haystack.document_stores import MilvusDocumentStore
    # See parameters required for index_type in ...
    # There are two kinds of parameters: ones required at construction/indexing time, and ones required at search time.

    # IVF_FLAT => nlist | nprobe
    # HNSW => M, efConstruction | ef

    # Info related to IVF_FLAT:
    # nlist is set at the indexing time and nprobe is set at the search time (https://milvus.io/docs/index.md).
    # A bug in milvus store makes passing index_param necessary (https://github.com/deepset-ai/haystack/issues/4681)
    # Check this out to see how to set the nlist and nprobe parameters:
    # https://milvus.io/docs/v1.1.1/performance_faq.md#How-to-set-nlist-and-nprobe-for-IVF-indexes
    # The nlist=16348 and nprob=512 values are set based on the curve given there. May be revisit it at some point.
    # On increasing index_file_size, indexing will slow, but querying will be fast. https://milvus.io/docs/v1.1.1/tuning.md

    # Info related to HNSW:
    # DPR paper has used this: 512, 200 | 128 (see footnote 10)
    # Haystack benchmark uses (https://haystack.deepset.ai/benchmarks/v0.9.0): 128, 80 | 20
    # milvus caps M to be maximum 64, so I can't increase it beyond that.
    if index_type == "IVF_FLAT":
        index_param = {"nlist": 16384}
        search_param = {"nprobe": 512}
    elif index_type == "HNSW":
        index_param = {"M": 64, "efConstruction": 200}
        search_param = {"ef": 128}
    else:
        raise Exception(f"index_type must be IVF_FLAT or HNSW, found {index_type}")

    document_store = MilvusDocumentStore(
        sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
        host=milvus_host, port=milvus_port,
        index=index_name, index_type=index_type,
        embedding_dim=768, id_field="id", embedding_field="embedding",
        progress_bar=True,
        index_file_size=1024,
        index_param=index_param,
        search_param={"params": search_param}, # yeah, there's a bug due to which "params" is also needed!
        similarity="dot_product",
    )
    return document_store
