from sqlalchemy import create_engine, MetaData, Table, Column, String, UniqueConstraint, inspect
from sqlalchemy.sql import text, select
import pandas as pd


class HLDatabase:
    def __init__(self,
                 db_path: str = None,
                 bWorkLaptop: bool = False
                 ):

        if db_path is None and bWorkLaptop is True:
            self.path = r'sqlite:///C:\Users\VY72PC\PycharmProjects\Academia\Thesis-Code\HollandseLuchten.sqlite'
        elif db_path is None and bWorkLaptop is False:
            self.path = r"sqlite:////Users/main/Vault/Thesis/Code/HollandseLuchten.sqlite"
        else:
            self.path = db_path
        self.engine = create_engine(self.path)
        self.connection = self.engine.connect()
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)

        self.metadata_table = Table(
            'metadata',
            self.metadata,
            Column('table_name', String, nullable=False),
            Column('geo_level', String, nullable=False),
            Column('time_level', String, nullable=False),
            UniqueConstraint('table_name', 'geo_level', 'time_level', name='uix_1')
        )

        if not self.inspector.has_table("metadata"):
            self.metadata_table.create(self.engine)

    def clear_database(self):
        with self.engine.connect() as conn:
            conn.execute(text('DROP TABLE IF EXISTS metadata'))
            conn.execute(text('PRAGMA foreign_keys = ON'))
            for table in self.metadata.tables.values():
                conn.execute(text(f'DROP TABLE IF EXISTS {table.name}'))

    def add_dataframe(self, table_name, geo_level, time_level, df):
        df.to_sql(table_name, self.engine, if_exists='replace')

        with self.engine.connect() as conn:
            conn.execute(
                self.metadata_table.insert(),
                table_name=table_name,
                geo_level=geo_level,
                time_level=time_level
            )

    def get_table(self, table_name):
        return pd.read_sql(table_name, self.engine)

    def get_dataframe(self, geo_level, time_level):
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
