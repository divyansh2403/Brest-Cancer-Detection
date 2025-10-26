# data_pipeline/load_data.py
from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_wisconsin_as_df():
    """
    Loads the Breast Cancer Wisconsin dataset from sklearn
    and returns it as a pandas DataFrame.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = malignant, 1 = benign
    return df

# Test run
if __name__ == "__main__":
    df = load_wisconsin_as_df()
    print("Shape of dataset:", df.shape)
    print(df.head())
