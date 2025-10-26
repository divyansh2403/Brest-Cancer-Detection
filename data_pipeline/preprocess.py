# data_pipeline/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def basic_preprocess(df, drop_low_variance=True, scaler=None):
    """
    Cleans and scales the dataset.
    - Removes low-variance features
    - Scales values using StandardScaler
    Works with datasets that may or may not have a 'target' column.
    """
    # Drop target column safely if present
    X = df.drop(columns=['target'], errors='ignore').copy()

    # Extract target if present
    y = df['target'].copy() if 'target' in df.columns else None

    # Step 1: Drop constant or near-constant features
    if drop_low_variance:
        sel = VarianceThreshold(threshold=1e-5)
        X = pd.DataFrame(sel.fit_transform(X),
                         columns=[f for f in X.columns[sel.get_support(indices=True)]])

    # Step 2: Scale the data (normalize)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y.values if y is not None else None, scaler

# Test run
if __name__ == "__main__":
    from data_pipeline.load_data import load_wisconsin_as_df
    df = load_wisconsin_as_df()
    X_scaled, y, scaler = basic_preprocess(df)
    print("Processed features shape:", X_scaled.shape)
    if y is not None:
        print("Target shape:", y.shape)
