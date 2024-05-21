from src.features.build_features import Data_Processing

data_obj = Data_Processing()
df = data_obj.preprocess_data()

print(df)