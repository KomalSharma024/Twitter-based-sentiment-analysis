import pickle
import xgboost as xgb
import os

# Set the path to the model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# Load the pickled XGBoost model
with open(os.path.join(MODEL_DIR, "model_xgb.pkl"), "rb") as f:
    booster = pickle.load(f)

# Save it as a JSON model
json_path = os.path.join(MODEL_DIR, "model_xgb.json")
booster.save_model(json_path)

print("✅ Model saved as JSON at:", json_path)
