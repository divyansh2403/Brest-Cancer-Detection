# data_pipeline/split_save.py

import os
import joblib
from sklearn.model_selection import train_test_split
from .load_data import load_wisconsin_as_df
from .preprocess import basic_preprocess


def prepare_and_save(test_size=0.2, random_state=42, out_dir='models'):
    # Load and preprocess data
    df = load_wisconsin_as_df()
    X, y, scaler = basic_preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save scaler and test data
    joblib.dump({'scaler': scaler}, f'{out_dir}/scaler.joblib')
    joblib.dump({'X_test': X_test, 'y_test': y_test}, f'{out_dir}/test_data.joblib')

    print('Saved scaler and test split to', out_dir)


if __name__ == '__main__':
    prepare_and_save()
