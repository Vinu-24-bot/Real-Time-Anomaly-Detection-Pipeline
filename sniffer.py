import time
import pandas as pd

def stream_data(input_csv="data/sample.csv", output_csv="data/stream.csv", delay=1):
    """
    Simulates real-time packet streaming by writing one row at a time.
    """
    df = pd.read_csv(input_csv)

    # Clear output file before streaming
    open(output_csv, "w").close()

    for i in range(len(df)):
        row = df.iloc[[i]]  # take one row
        if i == 0:
            row.to_csv(output_csv, mode="w", index=False)
        else:
            row.to_csv(output_csv, mode="a", header=False, index=False)
        
        print(f"Streamed row {i+1}/{len(df)}")
        time.sleep(delay)  # wait before sending next packet

if __name__ == "__main__":
    stream_data()