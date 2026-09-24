import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

iris = load_iris(as_frame=True)

df = iris.frame

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values Before")
print(df.isnull().sum())

df.loc[5, "sepal length (cm)"] = np.nan
df.loc[20, "petal width (cm)"] = np.nan

df["sepal length (cm)"] = df["sepal length (cm)"].fillna(
    df["sepal length (cm)"].median()
)

df["petal width (cm)"] = df["petal width (cm)"].fillna(
    df["petal width (cm)"].median()
)

print("\nMissing Values After")
print(df.isnull().sum())

encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nPreprocessed Training Data")
print(X_train.head())

print("\nTraining Labels")
print(y_train.head())