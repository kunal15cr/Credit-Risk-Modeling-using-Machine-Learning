from pathlib import Path
import joblib
label_encoder_obj_file_path = Path.cwd() / "data" / "artifacts" / "label_encoder.joblib"

obj = joblib.load(label_encoder_obj_file_path)

print(obj.inverse_transform( [1,3,3,2,1,3,3]))