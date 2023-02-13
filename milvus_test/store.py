from haystack.document_stores import MilvusDocumentStore

# mysql doesn't work due to some issues with sqlalchemy.
document_store = MilvusDocumentStore(
    sql_url="postgresql://postgres:postgres@127.0.0.1:5432/postgres"
)
document_store.write_documents([{
    "content": "sample text",
    "meta": {"title": "title", "text": "sample text"},
    "id_hash_keys": ["meta"]
}])
print(f"Num documents: {len(document_store.get_all_documents())}")

