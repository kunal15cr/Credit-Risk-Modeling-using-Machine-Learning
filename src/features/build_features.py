import sys
import pandas as pd
import numpy as np 
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import f_oneway
from src.logger import logging
from src.exception import CustomException
from scipy.stats import chi2_contingency as chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


@dataclass
class DataPath:
     internal_data_path = Path.cwd() / "data" / "Internal" / "internal.csv"
     external_data_path = Path.cwd() / "data" / "external" / "external.csv"
     train_csv_path  = Path.cwd() / "data" / "processed" / "train.csv"
     val_csv_path  = Path.cwd() / "data" / "processed" / "val.csv"

  
    



class Data_Processing:
    
    def __init__(self):
        self.all_path = DataPath()
        
    def load_data(self,data_path:str):
        return pd.read_csv(data_path)
    
    # this function chuck if feature has null value greater than 10000 or 20 this feature use less for any prediction
    # and send that feature list
    def null_size_chuck(self,df):
        try:
            columns_remove_list= []
            null_counts = df.eq(-99999).sum()
            columns_remove_list = null_counts[null_counts > 10000].index.tolist()
            return columns_remove_list
        except Exception as e:
            CustomException(e,sys)
            
            
        
        
    def remove_null(self,df): 
        try:
          return (
            (
                df
                .drop(self.null_size_chuck(df), axis=1)
                .replace(-99999, np.nan)
                .dropna()
            )
        ) 
        except Exception as e:
             CustomException(e,sys)
        
    def merge_columns(self,df1,df2,merge_column_on:str):
        try:
            df = pd.merge(df1,df2, how='inner',left_on= [merge_column_on],right_on=[merge_column_on])
        except Exception as e:
            CustomException(e,sys)
        return df
    
    
    def select_significant_features(self,df):
        cat_keep_columns = []
        cat_columns = df.select_dtypes(include=["O"]).columns.drop("Approved_Flag")
        
        for i in cat_columns:
            # Create a contingency table
            contingency_table = pd.crosstab(df[i], df['Approved_Flag'])
            
            # Perform chi-squared test
            chi2_stat, pval, _, _ = chi2(contingency_table)
            
            # Check p-value and decide whether to keep the column
            if pval <= 0.05:
                cat_keep_columns.append(i)
        
        return cat_keep_columns
        
    def vif_check(self,df):
        numerical_columns =  (df.select_dtypes(exclude="O")
                          .columns
                          .drop("PROSPECTID","Approved_Flag"))
        vif_df = df[numerical_columns]
        columns_to_be_kept = []
        columns_index = 0  
         
        for i in range(0, vif_df.shape[1]):
            vif_score = variance_inflation_factor(vif_df,columns_index)
            # print( i,"---" ,vif_score)
            
            if vif_score <= 6:
                columns_to_be_kept.append(numerical_columns[i])
                columns_index = columns_index+1
            else:
                vif_df = vif_df.drop([numerical_columns[i]],axis=1)
        return columns_to_be_kept
        
    
    def num_features(self,df):
        num_columns_to_kept = []
        
        for  i in self.vif_check(df):
            a = list(df[i])
            b = list(df["Approved_Flag"])
            
            group_P1 = [value for value, group in zip(a,b) if group == "P1"]
            group_P2 = [value for value, group in zip(a,b) if group == "P2"]
            group_P3 = [value for value, group in zip(a,b) if group == "P3"]
            group_P4 = [value for value, group in zip(a,b) if group == "P4"]
            
            f_statistic , p_value  = f_oneway(group_P1,group_P2,group_P3,group_P4)
   
            if p_value <= 0.5:
                num_columns_to_kept.append(i)
        return num_columns_to_kept
    
    def split_data(self,df,target,random_state:int,test_size:float):
        X = df.drop(columns=[target],axis=1)
        Y = df[target]
        x_tran,x_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,random_state=random_state)
        

        train_df = pd.concat([x_tran, y_train], axis=1)
        val_df = pd.concat([x_test, y_test], axis=1)
        train_df.to_csv(self.all_path.train_csv_path,index=False,header=True)
        val_df.to_csv(self.all_path.val_csv_path,index=False,header=True)
        return train_df,val_df
    
    
                    
        
                
    
    def preprocess_data(self):
        df1 = self.load_data(self.all_path.internal_data_path)
        df2 = self.load_data(self.all_path.external_data_path)
        df1 = self.remove_null(df1)
        df2 = self.remove_null(df2)
       
        
        df = self.merge_columns(df1,df2,"PROSPECTID")
    
        cat_feature = self.select_significant_features(df)
    
        num_feturs = self.num_features(df)
        
        all_feature = list(cat_feature) + list(num_feturs) + ["Approved_Flag"]
            
        df = df[all_feature]
        
        self.split_data(df,"Approved_Flag",43,0.2)
        
        return df
            
    
        
        
def main():
    data_obj = Data_Processing()
    data_obj.preprocess_data()
    
    

    

if __name__ == "__main__":
    main()