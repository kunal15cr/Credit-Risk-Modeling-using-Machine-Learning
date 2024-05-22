import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
class  DataTransformationConfig:
    preprocessor_obj_file_path = Path.cwd() / "data" / "artifacts" / "preprocessor.joblib"
    label_encoder_obj_file_path = Path.cwd() / "data" / "artifacts" / "label_encoder.joblib"
    

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def custom_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        education_mapping = {
            'SSC': 1,
            '12TH': 2,
            'GRADUATE': 3,
            'POST-GRADUATE': 4,
            'UNDER GRADUATE': 3,
            'OTHERS': 1,
            'PROFESSIONAL': 3
        }
        df['EDUCATION'] = df['EDUCATION'].map(education_mapping)
        return df

    def get_data_transformer_object(self):
        cat_columns = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
        ordinal_columns = ["EDUCATION"]
        target_columns = "Approved_Flag"

        # Define the transformers for the pipelines
        cat_pipeline = Pipeline(steps=[
            ("one_hot_encoder", OneHotEncoder())
        ])

        ordinal_pipeline = Pipeline(steps=[
            ("function_transformer", FunctionTransformer(func=self.custom_ordinal_encoding))
        ])
        
        

        # Combine the pipelines using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat_pipeline", cat_pipeline, cat_columns),
                ("ordinal_pipeline", ordinal_pipeline, ordinal_columns)
                
            ],
            remainder='passthrough'
        )

        return preprocessor

    def initiate_data_transformation(self, train_path: str, test_path: str):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessor_obj = self.get_data_transformer_object()
        label_encoder = LabelEncoder()

        target_columns = ["Approved_Flag"]

        input_features_train_df = train_df.drop(columns=target_columns, axis=1)
        target_feature_train_df = train_df[target_columns]
        target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
        classes = label_encoder.classes_
        print(classes)
        save_obj(self.data_transformation_config.label_encoder_obj_file_path ,label_encoder)
        

        input_feature_test_df = test_df.drop(columns=target_columns, axis=1)
        target_feature_test_df = test_df[target_columns]
        target_feature_test_df = label_encoder.transform(target_feature_test_df)

        input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
        input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

        train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        save_obj(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_obj)
        return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path