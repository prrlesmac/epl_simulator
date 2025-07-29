import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_postgres_engine():
    """Load DB credentials from .env and return a SQLAlchemy engine."""
    load_dotenv()

    db_creds_json = os.getenv("DB_CREDS")
    if not db_creds_json:
        raise ValueError(f"Missing DB_CREDS environment variable, {db_creds_json}")
    db_creds = json.loads(db_creds_json)
    user = db_creds["DB_USER"]
    password = db_creds["DB_PASSWORD"]
    host = db_creds["DB_HOST"]
    dbname = db_creds["DB_NAME"]
    port = db_creds["DB_PORT"]

    if not all([user, password, host, port, dbname]):
        raise ValueError(
            f"Missing one or more required DB environment variables. {db_creds}"
        )

    conn_str = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(conn_str)
