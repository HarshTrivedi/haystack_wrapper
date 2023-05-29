from lib import get_postgresql_address, get_milvus_address, load_cwd_dotenv


def test_postgresql_connection():

    from sqlalchemy import create_engine

    postgresql_host, postgresql_port = get_postgresql_address()

    try:
        engine = create_engine(f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres")
        engine.connect()
        print("Postgresql connection successful.")
    except:
        print("Postgresql connection failed.")


def test_milvus_connection():

    from pymilvus import connections

    milvus_host, milvus_port = get_milvus_address()

    try:
        connections.add_connection(default={"host": milvus_host, "port": milvus_port})
        connections.connect()
        print("Milvus connection successful.")
    except:
        print("Milvus connection failed.")


def main():
    load_cwd_dotenv()
    test_postgresql_connection()
    test_milvus_connection()


if __name__ == "__main__":
    main()
