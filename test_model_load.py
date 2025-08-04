import joblib

try:
    #model = joblib.load('app/models/xgboost_sleep_model.pkl')
    model = joblib.load('app/models/lgbm_sleep_model.pkl')
    print("Model successfully loaded")
except Exception as e:
    print("Error loading model", e)