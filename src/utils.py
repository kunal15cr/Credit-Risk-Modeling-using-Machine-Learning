import sys
import os 
from src.logger import logging
from src.exception import CustomException
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import pymysql


load_dotenv()

host= os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")





def read_sql_data(table_name:str):
    logging.info("Read sql database ")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established")
        df = pd.read_sql_query(f'SELECT * FROM bank_data.{table_name}',mydb)
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
    