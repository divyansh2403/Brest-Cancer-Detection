from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()

# Convert to a DataFrame for easy viewing
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = Malignant, 1 = Benign

# View first 5 rows
print(df.head())

# View basic info
print(df.info())

# View statistics
print(df.describe())
