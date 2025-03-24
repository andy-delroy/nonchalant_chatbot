# manages the database connection and session creation
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


DATABASE_URL = "mysql+pymysql://root:root@localhost/new_ams_andy"

#create an engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# attempt to connect and execute a simple query
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Connection successful:", result.scalar())
except Exception as e:
    print("Connection failed:", e)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
