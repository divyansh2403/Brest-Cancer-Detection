# api/predict_client.py
import requests
from data_pipeline.load_data import load_wisconsin_as_df
from data_pipeline.preprocess import basic_preprocess

def send_sample(url='http://127.0.0.1:5000/predict'):
    """
    Sends the first sample from the Wisconsin dataset to the prediction API.
    """
    # Load and preprocess data
    df = load_wisconsin_as_df()
    X, y, scaler = basic_preprocess(df)

    # Take first sample
    sample = X[0].tolist()

    # Send POST request
    resp = requests.post(url, json={'features': sample})
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")

if __name__ == '__main__':
    send_sample()
