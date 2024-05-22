import sys
import os 
from joblib import dump , load
import joblib
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
    

def save_obj(file_path:Path,obj: object):
    try:
        Path.mkdir(file_path.parent,exist_ok=True)
        joblib.dump(obj,file_path)
    except Exception as e:
        raise CustomException(e,sys)