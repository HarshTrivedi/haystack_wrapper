import os
from dotenv import load_dotenv


def test_postgresql_connection():

    from sqlalchemy import create_engine

    # Note the address shouldn't have http://
    # Note the default psql port is 5432 but I changed it to 8432 to avoid conflict on beaker.
    postgresql_host, postgresql_port = os.environ.get("POSTGRESQL_SERVER_ADDRESS", "localhost:5432").split(":")

    try:
        engine = create_engine(f"postgresql://postgres:postgres@{postgresql_host}:{postgresql_port}/postgres")
        engine.connect()
        print("Postgresql connection successful.")
    except:
        print("Postgresql connection failed.")


def test_milvus_connection():

    from pymilvus import connections

    # Note the address shouldn't have http://
    milvus_host, milvus_port = os.environ.get("MILVUS_SERVER_ADDRESS", "localhost:19530").split(":")

    try:
        connections.add_connection(default={"host": milvus_host, "port": milvus_port})
        connections.connect()
        print("Milvus connection successful.")
    except:
        print("Milvus connection failed.")


def main():
    load_dotenv()
    test_postgresql_connection()
    test_milvus_connection()


if __name__ == "__main__":
    main()
