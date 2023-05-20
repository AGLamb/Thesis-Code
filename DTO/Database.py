from sqlalchemy import create_engine, MetaData, Table, Column, String, UniqueConstraint, inspect
from sqlalchemy.sql import text, select
import pandas as pd


class HLDatabase:
    def __init__(self, db_path: str = "sqlite:////Users/main/Vault/Thesis/Code/HollandseLuchten.sqlite"):
        self.engine = create_engine(db_path)
        self.connection = self.engine.connect()
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)

        # Define table to hold metadata
        self.metadata_table = Table(
            'metadata',
            self.metadata,
            Column('table_name', String, nullable=False),
            Column('geo_level', String, nullable=False),
            Column('time_level', String, nullable=False),
            UniqueConstraint('table_name', 'geo_level', 'time_level', name='uix_1')
        )

        # Create metadata table if it doesn't exist
        if not self.inspector.has_table("metadata"):
            self.metadata_table.create(self.engine)

    def clear_database(self):
        # Clear the database
        with self.engine.connect() as conn:
            conn.execute(text('DROP TABLE IF EXISTS metadata'))
            conn.execute(text('PRAGMA foreign_keys = ON'))
            for table in self.metadata.tables.values():
                conn.execute(text(f'DROP TABLE IF EXISTS {table.name}'))

    def add_dataframe(self, table_name, geo_level, time_level, df):
        # Add DataFrame to the database with the provided table name
        df.to_sql(table_name, self.engine, if_exists='replace')

        # Add entry to metadata table
        with self.engine.connect() as conn:
            conn.execute(
                self.metadata_table.insert(),
                table_name=table_name,
                geo_level=geo_level,
                time_level=time_level
            )

    def get_table(self, table_name):
        # Retrieve a DataFrame by its specific table name
        return pd.read_sql(table_name, self.engine)

    def get_dataframe(self, geo_level, time_level):
        # Query metadata table for matching table name
        with self.engine.connect() as conn:
            result = conn.execute(
                select(
                    self.metadata_table.c.table_name
                ).where(
                    self.metadata_table.c.geo_level == geo_level
                ).where(
                    self.metadata_table.c.time_level == time_level
                ))
            table_name = result.fetchone()

            if table_name is not None:
                return pd.read_sql(table_name[0], self.engine)
            else:
                raise ValueError(f'Table with geo_level={geo_level}, time_level={time_level} does not exist.')

    def get_all_tables(self, geo_level, time_level):
        # Query metadata table for all matching table names
        with self.engine.connect() as conn:
            result = conn.execute(
                select(
                    self.metadata_table.c.table_name
                ).where(
                    self.metadata_table.c.geo_level == geo_level
                ).where(
                    self.metadata_table.c.time_level == time_level
                ))
            table_names = [row[0] for row in result.fetchall()]
            return table_names
