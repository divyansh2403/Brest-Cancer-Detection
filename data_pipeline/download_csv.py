import joblib

# Load your saved model object
obj = joblib.load("models/svm_joblib.pkl")  # path to your saved model
scaler = obj['scaler']

# Check which features the scaler was trained on
print(scaler.feature_names_in_)
