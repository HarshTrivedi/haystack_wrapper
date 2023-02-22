

def test_postgresql_connection():

    from sqlalchemy import create_engine

    # postgresql_host = "localhost"
    # postgresql_port = "5432"

    postgresql_host = "bore.pub"
    postgresql_port = "39549"

    try:
        engine = create_engine(f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres")
        engine.connect()
        print("Postgresql connection successful.")
    except:
        print("Postgresql connection failed.")


def test_milvus_connection():

    from pymilvus import connections

    #milvus_host = "localhost"
    #milvus_port = "19530"

    milvus_host = "bore.pub"
    milvus_port = "39863"

    try:
        connections.add_connection(default={"host": milvus_host, "port": milvus_port})
        connections.connect()
        print("Milvus connection successful.")
    except:
        print("Milvus connection failed.")


def main():
    test_postgresql_connection()
    test_milvus_connection()


if __name__ == "__main__":
    main()
