import json
import time
import argparse
import pandas as pd
from confluent_kafka import Producer

def produce(csv_path, bootstrap, topic, delay=0.2):
    df = pd.read_csv(csv_path)
    p = Producer({"bootstrap.servers": bootstrap})
    for i, row in df.iterrows():
        rec = row.to_dict()
        rec["ts_produce"] = time.time()
        payload = json.dumps(rec).encode("utf-8")
        p.produce(topic, value=payload)
        if i % 100 == 0:
            p.flush()
        print(f"sent {i+1}/{len(df)}")
        time.sleep(delay)
    p.flush()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test_network_data_100.csv")
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--topic", default="packets")
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()
    produce(args.csv, args.bootstrap, args.topic, args.delay)
