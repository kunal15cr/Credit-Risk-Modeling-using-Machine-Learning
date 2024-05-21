
import sys
import os 
import pymysql
from src.logger import logging
from src.exception import CustomException
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.utils import read_sql_data

from dataclasses import dataclass

@dataclass
class DataPath:
    internal_data_path = Path.cwd() / "data" / "Internal"
    external_data_path = Path.cwd() / "data" / "external"

# D:\project resume\Classification ML\credit_risk_modeling_using_machine_learning\data\external
# df = read_sql_data("case_study2")

class DataExtractSQL:
    
    def __init__(self) -> None:
        self.all_path = DataPath()
        
    def Data_store(self):
        logging.info("store internal vs external data")  
        try:
            internal_df = read_sql_data("case_study1")
            external_df = read_sql_data("case_study2")
            
            internal_df.to_csv(self.all_path.internal_data_path / "internal.csv",index=False,header=True)
            external_df.to_csv(self.all_path.external_data_path / "external.csv",index=False,header=True)
            
            return internal_df ,  external_df
        except Exception as e:
            CustomException(e,sys)
            
        
    
def main():
    data = DataExtractSQL()
    data.Data_store()
    
if __name__ == "__main__":
    main()