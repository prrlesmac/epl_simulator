import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_postgres_engine():
    """Load DB credentials from .env and return a SQLAlchemy engine."""
    load_dotenv()

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME")

    if not all([user, password, host, port, dbname]):
        raise ValueError("Missing one or more required DB environment variables.")

    conn_str = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(conn_str)
