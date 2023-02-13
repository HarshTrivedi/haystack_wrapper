from haystack.document_stores import MilvusDocumentStore

document_store = MilvusDocumentStore(
    sql_url="postgresql://postgres:postgres@127.0.0.1:5432/postgres"
)
print(f"Num documents: {len(document_store.get_all_documents())}")

