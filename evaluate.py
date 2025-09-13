import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("test_network_data_100.csv")
X = df.drop("label", axis=1)
y_true = df["label"]

# Load models
isolation_forest = joblib.load("models/isolation_forest.pkl")
ocsvm = joblib.load("models/ocsvm.pkl")
lof = joblib.load("models/lof.pkl")
autoencoder = load_model("models/autoencoder.keras", compile=False)
trained_columns = joblib.load("models/trained_columns.pkl")

# Align columns
for c in trained_columns:
    if c not in X.columns:
        X[c] = 0
X = X[trained_columns]

# Preprocess
preprocessor = isolation_forest.named_steps["preprocessor"]
X_proc = preprocessor.transform(X)
X_arr = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc

# Ensemble predictions
preds_list = []
preds_list.append(isolation_forest.predict(X))
preds_list.append(ocsvm.predict(X))
preds_list.append(lof.predict(X))

ae_recon = autoencoder.predict(X_arr, verbose=0)
mse = np.mean(np.square(X_arr - ae_recon), axis=1)
thr = np.percentile(mse, 95)    # Balanced threshold
preds_list.append(np.where(mse > thr, -1, 1))

preds = np.array(preds_list)
final_preds = []
for i in range(preds.shape[1]):
    v = preds[:, i]
    anom = np.sum(v == -1)
    norm = np.sum(v == 1)
    final_preds.append(1 if norm > anom else -1)

# Convert (-1,1) to labels (0=Normal,1=Anomaly)
y_pred = [0 if p == 1 else 1 for p in final_preds]

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=3))

# Calculate accuracy
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nAccuracy: {accuracy:.4f}")