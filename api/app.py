import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import random
from data_pipeline.load_data import load_wisconsin_as_df
from data_pipeline.preprocess import basic_preprocess

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/svm_joblib.pkl')
obj = joblib.load(MODEL_PATH)
model = obj['model']
scaler = obj['scaler']

# Features expected by the scaler/model
EXPECTED_FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'compactness error', 'concavity error', 'concave points error',
    'symmetry error', 'worst radius', 'worst texture', 'worst perimeter',
    'worst area', 'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]
EXPECTED_FEATURES_LOWER = [f.lower() for f in EXPECTED_FEATURES]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Please provide features array'}), 400

    features = np.array(data['features']).reshape(1, -1)
    if features.shape[1] != len(EXPECTED_FEATURES):
        return jsonify({
            'error': f'Expected {len(EXPECTED_FEATURES)} features, got {features.shape[1]}'
        }), 400

    features_scaled = scaler.transform(features)
    pred = int(model.predict(features_scaled)[0])
    proba = float(model.predict_proba(features_scaled)[0, 1])

    return jsonify({
        'prediction': 'Malignant' if pred == 0 else 'Benign',
        'probability_benign': float(round(proba, 4))
    })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Please upload a CSV file'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check if columns match (ignore order)
    if sorted(df.columns) != sorted(EXPECTED_FEATURES_LOWER):
        return jsonify({
            'error': f'CSV columns do not match expected features.\nExpected: {EXPECTED_FEATURES}\nGot: {list(df.columns)}'
        }), 400

    try:
        # Reorder columns to match expected
        df = df[[col for col in EXPECTED_FEATURES_LOWER]]
        X_scaled = scaler.transform(df.values)
    except Exception as e:
        return jsonify({'error': f'Error processing CSV: {str(e)}'}), 500

    results = []
    for idx, row in enumerate(X_scaled):
        row_reshaped = row.reshape(1, -1)
        pred = int(model.predict(row_reshaped)[0])
        proba = float(model.predict_proba(row_reshaped)[0, 1])
        results.append({
            'row': idx + 1,
            'prediction': 'Malignant' if pred == 0 else 'Benign',
            'probability_benign': float(round(proba, 4))
        })

    return jsonify({'results': results})

@app.route('/sample', methods=['GET'])
def sample():
    df = load_wisconsin_as_df()
    X, _, _ = basic_preprocess(df.copy(), scaler=scaler)
    i = random.randint(0, len(X) - 1)
    sample_features = X[i].tolist()
    return jsonify({'features': sample_features, 'index': i})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
