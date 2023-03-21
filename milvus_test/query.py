import os
from dotenv import load_dotenv
from haystack.document_stores import MilvusDocumentStore
from pymilvus import list_collections, connections

load_dotenv()
milvus_address = os.environ.get("MILVUS_SERVER_ADDRESS", "localhost:19530")
assert ":" in milvus_address, "The address must have ':' in it."
milvus_host, milvus_port = milvus_address.split(":")
milvus_host = (
    milvus_host.split("//")[1] if "//" in milvus_host else milvus_host # it shouldn't have http://
)
postgresql_host, postgresql_port = os.environ.get("POSTGRESQL_SERVER_ADDRESS", "localhost:8432").split(":")

connections.add_connection(default={"host": milvus_host, "port": milvus_port})
connections.connect()

document_store = MilvusDocumentStore(
    sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
    host=milvus_host, port=milvus_port,
    # I can pass index=index_name if not default is needed (see list_collections())
)
print(f"Num documents: {len(document_store.get_all_documents())}")
