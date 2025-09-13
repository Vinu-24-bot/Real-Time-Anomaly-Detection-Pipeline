import os, sys, time, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from streamlit_autorefresh import st_autorefresh

# Optional libs
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None
try:
    import shap, matplotlib.pyplot as plt
except Exception:
    shap = None; plt = None

# Config
MODELS_DIR = "models"
FEEDBACK_PATH = os.path.join(MODELS_DIR, "feedback.csv")
STREAM_FILES = ["data/stream.csv", "data/streamed_data.csv"]
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
KAFKA_ANOMALIES_TOPIC = os.environ.get("KAFKA_ANOMALIES_TOPIC", "anomalies")

# Setup
st.set_page_config(page_title="AI Anomaly Detection", layout="wide")
st.title("AI-Powered Network Anomaly Detection — Kafka + Dataflow Ready")

# Model loaders
def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

def safe_load_keras(path):
    if load_model is None:
        st.error("TensorFlow/Keras not installed.")
        st.stop()
    try:
        return load_model(path, compile=False)
    except Exception as e:
        st.error(f"Failed to load Keras model {path}: {e}")
        st.stop()

# Check models
req = {
    "isolation_forest": os.path.join(MODELS_DIR, "isolation_forest.pkl"),
    "lof": os.path.join(MODELS_DIR, "lof.pkl"),
    "ocsvm": os.path.join(MODELS_DIR, "ocsvm.pkl"),
    "autoencoder": os.path.join(MODELS_DIR, "autoencoder.keras"),
    "trained_columns": os.path.join(MODELS_DIR, "trained_columns.pkl"),
    "preprocessor": os.path.join(MODELS_DIR, "preprocessor.pkl"),
}
for k, v in req.items():
    if not os.path.exists(v):
        st.warning(f"Missing {v}. Run training first.")

isolation_forest = safe_joblib_load(req["isolation_forest"]) if os.path.exists(req["isolation_forest"]) else None
lof = safe_joblib_load(req["lof"]) if os.path.exists(req["lof"]) else None
ocsvm = safe_joblib_load(req["ocsvm"]) if os.path.exists(req["ocsvm"]) else None
autoencoder = safe_load_keras(req["autoencoder"]) if os.path.exists(req["autoencoder"]) else None
trained_columns = safe_joblib_load(req["trained_columns"]) if os.path.exists(req["trained_columns"]) else []
preprocessor = safe_joblib_load(req["preprocessor"]) if os.path.exists(req["preprocessor"]) else None

# Ensure feedback exists
os.makedirs(MODELS_DIR, exist_ok=True)
if not os.path.exists(FEEDBACK_PATH):
    pd.DataFrame(columns=list(trained_columns) + ["label","timestamp"]).to_csv(FEEDBACK_PATH, index=False)

# Helpers
def _to_preprocessed_array(df):
    if preprocessor is None:
        st.error("No preprocessor found. Retrain first.")
        st.stop()
    Xp = preprocessor.transform(df)
    return Xp.toarray() if hasattr(Xp, "toarray") else Xp

def align_df(input_df):
    df = input_df.copy()
    # handle protocol_type unknowns
    if "protocol_type" in df.columns:
        df["protocol_type"] = df["protocol_type"].apply(lambda x: x if x in ["tcp","udp","icmp"] else "tcp")
    for c in trained_columns:
        if c not in df.columns:
            df[c] = 0
    extras = [c for c in df.columns if c not in trained_columns]
    if extras:
        df = df.drop(columns=extras)
    return df[trained_columns]

def ensemble_predict(df):
    preds_list = []
    if isolation_forest is not None:
        preds_list.append(isolation_forest.predict(df))
    if lof is not None:
        preds_list.append(lof.predict(df))
    if ocsvm is not None:
        preds_list.append(ocsvm.predict(df))
    if autoencoder is not None:
        X_arr = _to_preprocessed_array(df)
        ae_recon = autoencoder.predict(X_arr, verbose=0)
        mse = np.mean(np.square(X_arr - ae_recon), axis=1)
        thr = np.percentile(mse, 95)
        preds_list.append(np.where(mse > thr, -1, 1))
    if not preds_list:
        st.error("No models available.")
        st.stop()
    preds = np.array(preds_list)
    n_models = preds.shape
    final, confs = [], []
    for i in range(preds.shape[1]):
        v = preds[:, i]
        anom = np.sum(v == -1)
        norm = np.sum(v == 1)
        final.append(-1 if anom > norm else 1)
        confs.append(max(anom, norm) / n_models)
    return np.array(final), np.array(confs), preds

def compute_severity(df):
    try:
        iso_score = -isolation_forest.decision_function(df) if isolation_forest is not None else 0
        oc_score = -ocsvm.decision_function(df) if ocsvm is not None else 0
        lof_score = -lof.decision_function(df) if lof is not None else 0
        X_arr = _to_preprocessed_array(df)
        ae_recon = autoencoder.predict(X_arr, verbose=0)
        ae_mse = np.mean(np.square(X_arr - ae_recon), axis=1)
        def _norm01(x):
            import numpy as np
            x = np.asarray(x, dtype=float)
            lo, hi = np.nanmin(x), np.nanmax(x)
            if np.isclose(hi, lo): return np.zeros_like(x)
            return (x - lo) / (hi - lo + 1e-9)
        s = 0.30*_norm01(iso_score) + 0.25*_norm01(oc_score) + 0.20*_norm01(lof_score) + 0.25*_norm01(ae_mse)
        return (s * 100).round(1), ae_mse
    except Exception:
        return np.zeros(len(df)), np.zeros(len(df))

page = st.sidebar.radio("Navigation", ["Batch Detection", "Real-Time CSV", "Real-Time Kafka", "Explainability (SHAP)", "Feedback & Retrain"])

if page == "Batch Detection":
    st.header("Batch CSV Detection (Ensemble)")
    uploaded = st.file_uploader("Upload CSV (any schema will be aligned)", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.subheader("Original Data Preview")
        st.dataframe(df_raw.head())
        df = align_df(df_raw)
        st.subheader("Preview (aligned to training)")
        st.dataframe(df.head())
        if st.button("Run Detection"):
            preds, confs, votes = ensemble_predict(df)
            severity, ae_mse = compute_severity(df)
            out = df_raw.copy()
            out["Prediction"] = preds
            out["Confidence"] = confs.round(3)
            out["Severity(0-100)"] = severity
            out["LabelText"] = out["Prediction"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
            st.subheader(f"Results (all {len(out)} rows)")
            st.dataframe(out)
            st.markdown("### Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anomalies", int((preds == -1).sum()))
            c2.metric("Normals", int((preds == 1).sum()))
            c3.metric("Total Rows", len(out))
            c4.metric("Avg Severity", float(np.mean(severity)))
            st.download_button("Download results CSV", out.to_csv(index=False).encode("utf-8"), "anomaly_results.csv", "text/csv")

elif page == "Real-Time CSV":
    st.header("Real-Time Monitor (CSV fallback)")
    st.info("Run: python sniffer.py in another terminal. This page auto-refreshes every 3 seconds.")
    stream_file = next((f for f in STREAM_FILES if os.path.exists(f)), None)
    if stream_file is None:
        st.warning("No stream file found. Start sniffer.py to generate data/stream.csv.")
    else:
        st_autorefresh(interval=3000, key="rt_autorefresh_csv")
        try:
            df_stream_raw = pd.read_csv(stream_file)
            if df_stream_raw.empty:
                st.info("Stream file exists but is empty (waiting for rows).")
            else:
                df_stream = align_df(df_stream_raw)
                preds, confs, votes = ensemble_predict(df_stream)
                sev, ae_mse = compute_severity(df_stream)
                results = df_stream_raw.copy()
                results["Prediction"] = preds
                results["Confidence"] = confs.round(3)
                results["Severity(0-100)"] = sev
                results["LabelText"] = results["Prediction"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
                c1, c2, c3 = st.columns(3)
                c1.metric("Anomalies (snapshot)", int((results["Prediction"]==-1).sum()))
                c2.metric("Normals (snapshot)", int((results["Prediction"]==1).sum()))
                c3.metric("Avg Severity", float(results["Severity(0-100)"].mean()))
                st.markdown("### Latest packets (last 10)")
                st.dataframe(results.tail(10))
                chart_df = pd.DataFrame({
                    "anomaly_flag": (results["Prediction"] == -1).astype(int),
                    "normal_flag": (results["Prediction"] == 1).astype(int)
                }).cumsum()
                st.line_chart(chart_df[["anomaly_flag","normal_flag"]])
        except Exception as e:
            st.error(f"Error reading stream file: {e}")

elif page == "Real-Time Kafka":
    st.header("Real-Time Kafka Monitor")
    st.write(f"Bootstrap: {KAFKA_BOOTSTRAP} | Topic: {KAFKA_ANOMALIES_TOPIC}")
    try:
        from confluent_kafka import Consumer
    except Exception:
        st.error("confluent-kafka not installed. pip install confluent-kafka")
        st.stop()
    if "kafka_rows" not in st.session_state:
        st.session_state.kafka_rows = []
    st_autorefresh(interval=3000, key="rt_autorefresh_kafka")
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": "st-anomaly-ui",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([KAFKA_ANOMALIES_TOPIC])
    # poll a small batch per refresh
    for _ in range(200):
        msg = consumer.poll(0.01)
        if msg is None: break
        if msg.error(): continue
        data = json.loads(msg.value().decode("utf-8"))
        st.session_state.kafka_rows.append(data)
    consumer.close()
    rows = st.session_state.kafka_rows[-500:]
    if not rows:
        st.info("Waiting for anomalies...")
    else:
        df = pd.json_normalize(rows)
        df["is_anomaly"] = df["anomaly"].map({1:"Anomaly",0:"Normal"})
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalies (window)", int((df["anomaly"]==1).sum()))
        c2.metric("Normals (window)", int((df["anomaly"]==0).sum()))
        c3.metric("Avg latency (ms)", int(df["latency_ms"].mean()))
        st.markdown("### Latest (last 20)")
        st.dataframe(df.tail(20))
        st.download_button("Download window JSON", df.to_json(orient="records").encode("utf-8"), "anomalies_window.json", "application/json")
        # Quick feedback on last 10
        st.subheader("Feedback (last 10)")
        fb_rows = []
        for i, rec in enumerate(rows[-10:]):
            st.write(f"Row {len(rows)-10+i+1}")
            st.json(rec)
            choice = st.radio(f"Label?", ["Normal","Anomaly"], index=1 if rec.get("anomaly",0)==1 else 0, key=f"kfb_{i}")
            fb_rows.append((rec, choice))
        if st.button("Save feedback rows"):
            to_save = []
            for rec, choice in fb_rows:
                feat = rec.get("features", {})
                row = {c: feat.get(c, 0) for c in trained_columns}
                row["label"] = 1 if choice == "Anomaly" else 0
                row["timestamp"] = pd.Timestamp.now().isoformat()
                to_save.append(row)
            fb_df = pd.DataFrame(to_save)
            old = pd.read_csv(FEEDBACK_PATH) if os.path.exists(FEEDBACK_PATH) else pd.DataFrame(columns=fb_df.columns)
            pd.concat([old, fb_df], ignore_index=True).to_csv(FEEDBACK_PATH, index=False)
            st.success(f"Saved {len(fb_df)} feedback rows.")

elif page == "Explainability (SHAP)":
    st.header("Explainability (SHAP)")
    if shap is None or plt is None:
        st.warning("Install shap and matplotlib to enable this feature.")
    else:
        shap_toggle = st.checkbox("Enable SHAP (small datasets only)", value=False)
        uploaded = st.file_uploader("Upload CSV for SHAP", type=["csv"])
        if shap_toggle and uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            df = align_df(df_raw)
            st.subheader("Data sample for SHAP")
            st.dataframe(df.head())
            with st.spinner("Computing SHAP..."):
                Xp = _to_preprocessed_array(df)
                try:
                    tree_model = isolation_forest.named_steps["model"]
                    expl = shap.TreeExplainer(tree_model)
                    shap_vals = expl.shap_values(Xp[:min(200, Xp.shape)])
                    st.subheader("SHAP summary (top features)")
                    shap.summary_plot(shap_vals, pd.DataFrame(Xp[:min(200, Xp.shape)]), show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")

else:
    st.header("Feedback & Retrain")
    st.info(f"Feedback path: {FEEDBACK_PATH}")
    uploaded = st.file_uploader("Upload CSV to label", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df = align_df(df_raw)
        preds, confs, votes = ensemble_predict(df)
        sev, _ = compute_severity(df)
        out = df_raw.copy()
        out["Prediction"] = preds
        out["Confidence"] = confs.round(3)
        out["Severity(0-100)"] = sev
        out["LabelText"] = out["Prediction"].apply(lambda x: "Anomaly" if x == -1 else
