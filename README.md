# Real-Time-Anomaly-Detection-Pipeline
 Designed a streaming anomaly detection system with Kafka + GCP Dataflow + TensorFlow inference

Real-Time Anomaly Detection Pipeline
A streaming anomaly detection system with Kafka ingestion, Apache Beam processing, TensorFlow autoencoder inference, and a Streamlit UI for monitoring and feedback‑driven retraining.

Features
Kafka producer streams packet records to an input topic; Beam reads, infers anomalies, and writes results to an output topic.

TensorFlow autoencoder inference in-stream with percentile thresholding for severity and anomaly flags.

Streamlit UI supports batch detection, CSV real-time fallback, real-time Kafka monitoring, SHAP explainability, and feedback capture.

Feedback‑driven retraining script refreshes models and preprocessor for continuous improvement.

Evaluation script prints confusion matrix, classification report, and accuracy; tune threshold to approach target F1.

Tech stack
Python, TensorFlow/Keras, scikit‑learn, Apache Beam, Kafka, Docker, Streamlit.

Project layout
app.py — Streamlit UI with Batch, Real‑Time CSV, Real‑Time Kafka, SHAP, and Feedback pages.

train_model.py — trains ensemble models, builds autoencoder, and saves preprocessor for consistent inference.

evaluate.py — computes metrics on the sample dataset.

retrain.py — retrains using labeled feedback rows and saves updated models.

sniffer.py — CSV streaming simulator for local fallback.

test_network_data_100.csv — sample dataset with label column.

data/stream.csv — sample streamed CSV for fallback real‑time view.

models/ — directory for saved models, preprocessor, and feedback.csv.

Quickstart (local)
Follow these steps to run the full local pipeline with Kafka, Beam (DirectRunner), and Streamlit.

1) Environment and training
text
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
This generates models/isolation_forest.pkl, models/ocsvm.pkl, models/lof.pkl, models/autoencoder.keras, models/trained_columns.pkl, and models/preprocessor.pkl.

2) Start Kafka with Docker
text
docker compose up -d
docker ps
Ensure Kafka and Zookeeper containers are running.

3) Create Kafka topics
text
# Find the Kafka container name (e.g., <repo>_kafka_1)
docker exec -it <kafka_container_name> \
  kafka-topics.sh --create --topic packets --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

docker exec -it <kafka_container_name> \
  kafka-topics.sh --create --topic anomalies --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

docker exec -it <kafka_container_name> \
  kafka-topics.sh --list --bootstrap-server localhost:9092
Input topic is packets and output topic is anomalies.

4) Start the producer (send data to Kafka)
text
python producer_kafka.py --csv test_network_data_100.csv --bootstrap 127.0.0.1:9092 --topic packets --delay 0.2
This streams JSON events with a ts_produce timestamp used for end‑to‑end latency measurement.

5) Run the Beam pipeline locally
text
python beam_pipeline.py \
  --runner DirectRunner \
  --bootstrap_servers 127.0.0.1:9092 \
  --input_topic packets \
  --output_topic anomalies \
  --model_uri models/autoencoder.keras \
  --preproc_uri models/preprocessor.pkl \
  --threshold_percentile 95.0 \
  --streaming
The pipeline reads from Kafka, runs TensorFlow autoencoder inference, computes mse/threshold/severity, and writes JSON results to anomalies.

6) Open the Streamlit UI
text
# Optional (defaults shown):
# Windows (CMD): set KAFKA_BOOTSTRAP=127.0.0.1:9092
# PowerShell: $env:KAFKA_BOOTSTRAP="127.0.0.1:9092"
# Linux/Mac: export KAFKA_BOOTSTRAP=127.0.0.1:9092

# Optional: set KAFKA_ANOMALIES_TOPIC=anomalies

streamlit run app.py
Use the Real‑Time Kafka page to see live anomalies, latency_ms, and quick feedback capture; use Real‑Time CSV as a fallback.

7) Save feedback and retrain
In the Real‑Time Kafka page, label recent rows and press “Save feedback rows” to append to models/feedback.csv.

When enough feedback is collected, retrain the stack and refresh models.

text
python retrain.py
After retraining, re‑run local Beam or re‑deploy to pick up updated models/preprocessor.

8) Evaluate metrics
text
python evaluate.py
This prints confusion matrix, classification report, and accuracy; adjust threshold_percentile in the pipeline to tune precision/recall.

Cloud deployment (GCP Dataflow)
Run the same Beam job managed by Dataflow with a custom SDK container that includes TensorFlow and dependencies.

1) Upload artifacts to GCS
text
gsutil mb -l <REGION> gs://<BUCKET>
gsutil cp models/autoencoder.keras gs://<BUCKET>/models/autoencoder.keras
gsutil cp models/preprocessor.pkl gs://<BUCKET>/models/preprocessor.pkl
These URIs are passed to the pipeline so workers can download them on startup.

2) Build and push the Beam worker image
text
# Example tag
docker build -t <REGION>-docker.pkg.dev/<PROJECT>/<REPO>/beam-tf:v1 -f Dockerfile.beam .
docker push <REGION>-docker.pkg.dev/<PROJECT>/<REPO>/beam-tf:v1
Pin versions and avoid latest tags for reliable streaming jobs.

3) Launch the Dataflow streaming job
text
python beam_pipeline.py \
  --runner DataflowRunner \
  --project <PROJECT> \
  --region <REGION> \
  --temp_location gs://<BUCKET>/tmp \
  --staging_location gs://<BUCKET>/staging \
  --sdk_container_image <REGION>-docker.pkg.dev/<PROJECT>/<REPO>/beam-tf:v1 \
  --bootstrap_servers <KAFKA_BROKER_DNS_OR_IP>:9092 \
  --input_topic packets \
  --output_topic anomalies \
  --model_uri gs://<BUCKET>/models/autoencoder.keras \
  --preproc_uri gs://<BUCKET>/models/preprocessor.pkl \
  --threshold_percentile 95.0 \
  --streaming
Ensure Dataflow workers can reach the Kafka brokers over the network or VPC peering.

Configuration
Environment variables used by the Streamlit UI.

KAFKA_BOOTSTRAP: Kafka bootstrap servers (default 127.0.0.1:9092).

KAFKA_ANOMALIES_TOPIC: Kafka topic read by the UI (default anomalies).

Key model files expected in models/.

isolation_forest.pkl, ocsvm.pkl, lof.pkl, autoencoder.keras, preprocessor.pkl, trained_columns.pkl.

feedback.csv — created automatically on first run if missing.

Datasets
test_network_data_100.csv — small labeled dataset for training, producing, and evaluation.

data/stream.csv — generated by sniffer.py for CSV fallback real‑time monitoring.

How it works
Producer publishes JSON packet records with a ts_produce field to Kafka topic packets.

Beam pipeline consumes packets, applies preprocessor.pkl, runs autoencoder.keras, computes mse and a percentile threshold, then outputs anomaly=1/0, severity, and latency_ms to anomalies.

Streamlit consumes anomalies in real time, shows metrics, allows quick labeling, and saves feedback.csv for retraining.

retrain.py refreshes models using feedback and writes updated artifacts to models/.

Troubleshooting
No anomalies appearing: confirm producer is running and topics exist; verify KAFKA_BOOTSTRAP and topic names match.

UI errors about missing models: re-run training to generate all model and preprocessor files under models/.

CSV fallback not updating: run sniffer.py to create/append data/stream.csv and open the Real‑Time CSV page.

Low F1: tune threshold_percentile in beam_pipeline.py and retrain with more labeled feedback.

Scripts and commands summary
Train: python train_model.py.

Produce: python producer_kafka.py --csv test_network_data_100.csv --bootstrap 127.0.0.1:9092 --topic packets.

Pipeline (local): python beam_pipeline.py --runner DirectRunner --bootstrap_servers 127.0.0.1:9092 --input_topic packets --output_topic anomalies --model_uri models/autoencoder.keras --preproc_uri models/preprocessor.pkl --streaming.

UI: streamlit run app.py.

Retrain: python retrain.py.

Evaluate: python evaluate.py.
