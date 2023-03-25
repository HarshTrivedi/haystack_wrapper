from dotenv import load_dotenv
from haystack.document_stores import MilvusDocumentStore
from pymilvus import connections
from lib import get_postgresql_address, get_milvus_address

load_dotenv()

milvus_host, milvus_port = get_milvus_address()
postgresql_host, postgresql_port = get_postgresql_address()

connections.add_connection(default={"host": milvus_host, "port": milvus_port})
connections.connect()

# mysql doesn't work due to some issues with sqlalchemy.
document_store = MilvusDocumentStore(
    sql_url=f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres",
    host=milvus_host, port=milvus_port
    # I can pass index=index_name if not default is needed (see list_collections())
)

document_store.write_documents([{
    "content": "sample text",
    "meta": {"title": "title", "text": "sample text"},
    "id_hash_keys": ["meta"]
}])
print(f"Num documents: {len(document_store.get_all_documents())}")
