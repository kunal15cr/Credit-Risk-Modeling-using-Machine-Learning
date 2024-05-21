
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,FunctionTransformer

df_path = Path.cwd() / "data" / "processed" / "train.csv"
df = pd.read_csv(df_path)

class 
