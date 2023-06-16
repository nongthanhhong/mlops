import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans
from utils import *
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from yellowbrick.cluster import KElbowVisualizer
from eda_data import DataAnalyzer
from raw_data_processor import *


def label_captured_data(prob_config: ProblemConfig, model_params):
    train_x = pd.read_parquet(prob_config.train_x_path).to_numpy()
    train_y = pd.read_parquet(prob_config.train_y_path).to_numpy()
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])

    eda = DataAnalyzer(prob_config)
    eda.data, _ = RawDataProcessor.build_category_features(
            captured_x, prob_config.categorical_cols
        )
    captured_x = eda.input_process()

    np_captured_x = captured_x.to_numpy()
    n_captured = len(np_captured_x)
    n_samples = len(train_x) + n_captured

    logging.info(f"Loaded {n_captured} captured samples, {n_samples} train + captured")

    logging.info("Initialize and fit the clustering model")
    
    model = MiniBatchKMeans()
    k_mean = int( len(train_y) / 100) * len(np.unique(train_y))
    
    # Use the KElbowVisualizer to find the optimal k using elbow method
    visualizer = KElbowVisualizer(model, k=(k_mean-5, k_mean + 5))
    visualizer.fit(train_x)
    visualizer.show()   
    optimal_k = visualizer.elbow_value_
    if optimal_k is None:
        optimal_k = k_mean
        print(optimal_k)
    else: 
        print(optimal_k)

    kmeans_model = MiniBatchKMeans(
        n_clusters=optimal_k, **model_params
    )
    kmeans_model.fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(np_captured_x)

    logging.info(
        "Assign new labels to the new data based on the labels of the original data in each cluster"
    )
    new_labels = []
    for i in range(optimal_k):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # For a linear regression problem, use the mean of the labels as the new label
            # For a logistic regression problem, use the mode of the labels as the new label
            if ml_type == "regression":
                new_labels.append(np.mean(cluster_labels.flatten()))
            else:
                new_labels.append(
                    np.bincount(cluster_labels.flatten().astype(int)).argmax()
                )

    approx_label = [new_labels[c] for c in kmeans_clusters]
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])

    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)


def merge_raw_captured(prob_config: ProblemConfig):

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])
    
    captured_x = captured_x[["feature4", "feature7", "feature3"]]

    # np_captured_x = captured_x.to_numpy()
    train_x = pd.read_parquet(prob_config.raw_data_path)[["feature4", "feature7", "feature3"]]

    dtype = train_x.dtypes.to_frame('dtypes').reset_index().set_index('index')['dtypes'].astype(str).to_dict()

    train_x = pd.DataFrame(np.concatenate((train_x, captured_x)), columns=["feature4", "feature7", "feature3"])
    
    for column in train_x.columns:
        train_x[column] = train_x[column].astype(dtype[column])
    
    train_x.to_parquet(prob_config.captured_x_path, index=False)
    logging.info(f"Saved {len(train_x)} samples to {prob_config.captured_x_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--config-path", type=str, default="./src/model_config/minibatchkmeans.yaml")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        model_params = yaml.safe_load(f)

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    # label_captured_data(prob_config, model_params)
    merge_raw_captured(prob_config)
