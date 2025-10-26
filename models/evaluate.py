# models/evaluate.py
import argparse
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt

def evaluate(model_path='models/svm_joblib.pkl'):
    # Load model and scaler
    obj = joblib.load(model_path)
    model = obj['model']
    scaler = obj['scaler']

    # Load test data if available
    try:
        td = joblib.load('models/test_data.joblib')
        X_test = td['X_test']
        y_test = td['y_test']
    except Exception:
        # fallback: recreate split deterministically
        from data_pipeline.load_data import load_wisconsin_as_df
        from data_pipeline.preprocess import basic_preprocess
        from sklearn.model_selection import train_test_split

        df = load_wisconsin_as_df()
        X, y, _ = basic_preprocess(df, scaler=scaler)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1:', f1_score(y_test, y_pred))
    
    try:
        print('ROC AUC:', roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n', cm)

    # Plot ROC curve
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.legend()
        plt.savefig('models/roc_curve.png')
        print('Saved ROC curve to models/roc_curve.png')
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', dest='model',
        default='models/svm_joblib.pkl',
        help='Path to the trained model file'
    )
    args = parser.parse_args()
    evaluate(args.model)
