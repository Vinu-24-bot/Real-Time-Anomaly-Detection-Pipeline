import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Load feedback
feedback = pd.read_csv("feedback.csv")

# Use only rows marked "Incorrect" for retraining
train_data = feedback[feedback["Feedback"] == "Incorrect"].drop(columns=["Feedback", "Prediction", "Confidence"])

# --- Get known categories from the original model ---
# Load the original preprocessor to get known categories
original_iso = joblib.load("models/isolation_forest.pkl")
preprocessor = original_iso.named_steps["preprocessor"]
cat_transformer = preprocessor.named_transformers_['cat']
known_categories = cat_transformer.categories_[0]

# Filter out unknown categories from the 'protocol_type' column in the feedback data
if 'protocol_type' in train_data.columns:
    train_data['protocol_type'] = train_data['protocol_type'].apply(lambda x: x if x in known_categories else known_categories[0])
# --- END OF ADDITION ---

# Retrain models (simple example, extendable)
iso = IsolationForest(contamination=0.1, random_state=42).fit(train_data)
ocsvm = OneClassSVM(nu=0.1, kernel="rbf").fit(train_data)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train_data)

# Save updated models
joblib.dump(iso, "models/isolation_forest.pkl")
joblib.dump(ocsvm, "models/ocsvm.pkl")
joblib.dump(lof, "models/lof.pkl")

print("✅ Models retrained successfully from feedback!")