import os
import io
import json
import time
import argparse
import joblib
import typing as t
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.io.kafka import ReadFromKafka, WriteToKafka
from tensorflow.keras.models import load_model
from google.cloud import storage

FEATURE_ORDER = ["duration","src_bytes","dst_bytes","protocol_type","count","srv_count"]

def download_from_gcs(uri: str, local_path: str):
    if not uri.startswith("gs://"):
        return uri
    client = storage.Client()
    _, bucket, *rest = uri.replace("gs://", "").split("/")
    blob_name = "/".join(rest)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

class InferenceDoFn(beam.DoFn):
    def __init__(self, model_uri: str, preproc_uri: str, thr_percentile: float):
        self.model_uri = model_uri
        self.preproc_uri = preproc_uri
        self.thr_percentile = thr_percentile

    def setup(self):
        model_path = download_from_gcs(self.model_uri, "/tmp/autoencoder.keras")
        preproc_path = download_from_gcs(self.preproc_uri, "/tmp/preprocessor.pkl")
        self.model = load_model(model_path, compile=False)
        self.preprocessor = joblib.load(preproc_path)
        self.mse_history = []

    def process(self, element: t.Tuple[bytes, bytes]):
        # element=(key,value) from Kafka; we use value only
        value = element[1]
        rec = json.loads(value.decode("utf-8"))
        ts_in = rec.get("ts_produce", time.time())
        # Build feature row
        row = {k: rec.get(k, 0) for k in FEATURE_ORDER}
        # Handle protocol_type unknowns
        if row["protocol_type"] not in ["tcp","udp","icmp"]:
            row["protocol_type"] = "tcp"
        # Preprocess
        import pandas as pd
        X_df = pd.DataFrame([row])
        Xp = self.preprocessor.transform(X_df)
        X_arr = Xp.toarray() if hasattr(Xp, "toarray") else Xp
        # TF inference
        recon = self.model.predict(X_arr, verbose=0)
        mse = float(((X_arr - recon)**2).mean())
        # Adaptive threshold (percentile over sliding buffer)
        self.mse_history.append(mse)
        hist = self.mse_history[-5000:]
        import numpy as np
        thr = float(np.percentile(hist, self.thr_percentile)) if len(hist) >= 20 else mse*1.2
        is_anom = int(mse > thr)
        severity = float(min(100.0, max(0.0, 100.0 * (mse / (thr + 1e-9)))))
        # Latency
        ts_out = time.time()
        latency_ms = int((ts_out - ts_in) * 1000)
        out = {
            "features": row,
            "mse": mse,
            "threshold": thr,
            "anomaly": is_anom,
            "severity": severity,
            "ts_in": ts_in,
            "ts_out": ts_out,
            "latency_ms": latency_ms,
        }
        yield (b"", json.dumps(out).encode("utf-8"))

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap_servers", default="127.0.0.1:9092")
    ap.add_argument("--input_topic", default="packets")
    ap.add_argument("--output_topic", default="anomalies")
    ap.add_argument("--model_uri", default="models/autoencoder.keras")
    ap.add_argument("--preproc_uri", default="models/preprocessor.pkl")
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--runner", default="DirectRunner")
    ap.add_argument("--project", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--temp_location", default=None)
    ap.add_argument("--staging_location", default=None)
    ap.add_argument("--streaming", action="store_true", default=True)
    args, beam_unk = ap.parse_known_args()

    options_dict = {
        "runner": args.runner,
        "streaming": True,
    }
    if args.project:
        options_dict["project"] = args.project
    if args.region:
        options_dict["region"] = args.region
    if args.temp_location:
        options_dict["temp_location"] = args.temp_location
    if args.staging_location:
        options_dict["staging_location"] = args.staging_location

    pipeline_options = PipelineOptions(flags=beam_unk, **options_dict)
    standard_options = pipeline_options.view_as(StandardOptions)
    standard_options.streaming = True

    with beam.Pipeline(options=pipeline_options) as p:
        msgs = (
            p
            | "ReadKafka" >> ReadFromKafka(
                consumer_config={
                    "bootstrap.servers": args.bootstrap_servers,
                    "auto.offset.reset": "latest",
                    "group.id": "beam-anomaly-consumer",
                },
                topics=[args.input_topic],
            )
        )

        preds = (
            msgs
            | "Infer" >> beam.ParDo(
                InferenceDoFn(
                    model_uri=args.model_uri,
                    preproc_uri=args.preproc_uri,
                    thr_percentile=args.threshold_percentile,
                )
            )
        )

        _ = (
            preds
            | "WriteKafka" >> WriteToKafka(
                producer_config={"bootstrap.servers": args.bootstrap_servers},
                topic=args.output_topic,
            )
        )

if __name__ == "__main__":
    run()
