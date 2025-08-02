import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model = joblib.load('app/models/xgboost_sleep_model.pkl')

test_input = np.array([[1, 27, 9, 6.1, 42, 6, 3, 77, 4200, 2, 126, 83]])

prediction = model.predict(test_input)

pred_qual = prediction[0] + 4 #shifting right as model shifts values left

print(f'Predicted sleep quality: {pred_qual}')