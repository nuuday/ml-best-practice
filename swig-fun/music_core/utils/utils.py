import os
import getpass
import pandas as pd
import sys

from tqdm import tqdm
from ai_utils.sql import create_engine


def load_data(DEBUGGING, sql_query="select top 1000 * from DM.PA.Telmore_ABT_Latest"):
    """Example of how to load data.
    Change sql_query variable to whatever is needed.

    Arguments
    ---------
    DEBUGGING: bool
        Set to false when running script from commandline, otherwise true
    sql_query: string (optional)
        The query string to run against the SQL database

    Returns
    -------
    data: pandas.DataFrame
       loaded data
    """
    if DEBUGGING:
        user = input("username (use M-number): ")
        password = getpass.getpass("password: ")
    else:
        user = os.getenv("CREDENTIALS_SQL_USR")
        password = os.getenv("CREDENTIALS_SQL_PSW")

    SQL_SERVER_HOST = '10.74.133.90'
    SQL_SERVER_PORT = '2501'
    SQL_SERVER_DB = 'DM'
    SQL_SERVER_UID = r'accdom01\{}'.format(user)
    SQL_SERVER_PWD = password

    engine = create_engine(SQL_SERVER_HOST, SQL_SERVER_PORT, SQL_SERVER_DB, SQL_SERVER_UID, SQL_SERVER_PWD)

    df_load = pd.read_sql(sql_query, engine, chunksize=20000)
    data = pd.concat([chunk for chunk in tqdm(df_load, desc='Loading data', file=sys.stdout)], ignore_index=True)

    return data
