import sys

import pandas as pd


def show_parquet(path: str):
    df = pd.read_parquet(path=path)
    print(df)


# Usage: python utils/show_parquet.py /mnt/e/mlops-marathon/mlops-mara-sample-public/data/raw_data/phase-1/prob-1/raw_train.parquet
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        show_parquet(sys.argv[1])
    else:
        print("missing path")
        exit(1)
