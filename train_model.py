import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("test_network_data_100.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Columns
categorical_cols = ["protocol_type"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Traditional models
isolation_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", IsolationForest(n_estimators=100, contamination=0.1, random_state=42)),
])
ocsvm = Pipeline([
    ("preprocessor", preprocessor),
    ("model", OneClassSVM(kernel="rbf", nu=0.1)),
])
lof = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)),
])

print("Training Isolation Forest...")
isolation_forest.fit(X)
print("Training One-Class SVM...")
ocsvm.fit(X)
print("Training Local Outlier Factor...")
lof.fit(X)

# Autoencoder
print("Training Autoencoder...")
X_proc = preprocessor.fit_transform(X)
X_arr = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc
input_dim = X_arr.shape[1]
autoencoder = Sequential([
    Dense(32, activation="relu", input_shape=(input_dim,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    Dense(input_dim, activation="linear"),
])
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
autoencoder.fit(X_arr, X_arr, epochs=10, batch_size=32, shuffle=True, verbose=0)

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(isolation_forest, "models/isolation_forest.pkl")
joblib.dump(ocsvm, "models/ocsvm.pkl")
joblib.dump(lof, "models/lof.pkl")
autoencoder.save("models/autoencoder.keras", include_optimizer=False)
joblib.dump(X.columns.tolist(), "models/trained_columns.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Saved models and preprocessor to models/")
