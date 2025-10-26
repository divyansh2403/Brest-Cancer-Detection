# models/train.py
import argparse
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from data_pipeline.load_data import load_wisconsin_as_df
from data_pipeline.preprocess import basic_preprocess

def train_and_save(out_path='models/svm_joblib.pkl'):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Load and preprocess data
    df = load_wisconsin_as_df()
    X, y, scaler = basic_preprocess(df)

    # SVM with grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }

    svc = SVC(probability=True)
    grid = GridSearchCV(svc, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)

    print('Best params:', grid.best_params_)
    print('Best CV score:', grid.best_score_)

    # Save model and scaler together
    joblib.dump({
        'model': grid.best_estimator_,
        'scaler': scaler,
        'cv_results': grid.cv_results_
    }, out_path)

    print('Saved model to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out', '--out_path', dest='out_path', 
        default='models/svm_joblib.pkl', 
        help='Path to save the trained model'
    )
    args = parser.parse_args()
    train_and_save(args.out_path)
