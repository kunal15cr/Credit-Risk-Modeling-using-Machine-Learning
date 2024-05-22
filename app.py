from src.features.data_transformation import DataTransformation
from pathlib import Path



obj_data = DataTransformation()
train_path = Path.cwd() / "data" / "processed" / "train.csv"
test_path = Path.cwd() / "data" / "processed" / "val.csv"

print(train_path)
train,test,obj = obj_data.initiate_data_transformation(train_path,test_path)

print(len(train))
print(len(test))
print(obj)