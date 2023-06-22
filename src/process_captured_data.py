
import yaml
import json
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from utils import *
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from data_engineering import DataAnalyzer, FeatureExtractor
from raw_data_processor import *


def propagate_labels(labeled_data, labeled_labels, unlabeled_data):


    
    config_path = './src/model_config/'+ args.phase_id + '/' + args.prob_id +'/cluster.json'
    with open(config_path, "r") as f:
        model_params = json.load(f)

    algorithm = model_params["algorithm"]["name"] # 'k-means' or 'DBSCAN' or 'MiniBatchKMeans'
    logging.info(f"Use {algorithm} algorithm to labeling captured data")

    if algorithm == 'DBSCAN':
        # Step 2: Cluster the data using DBSCAN
        logging.info(f"Parameters: {model_params['dbscan']}")
        clusterer = DBSCAN(**model_params["dbscan"])

    elif algorithm == 'k-means':
        # Step 2: Cluster the data using k-means
        logging.info(f"Parameters: {model_params['k_means']}")
        clusterer = KMeans(**model_params["k_means"])

    elif algorithm == 'MiniBatchKMeans':
        # Step 2: Cluster the data using MiniBatchKMeans
        logging.info(f"Parameters: {model_params['mini']}")
        clusterer = MiniBatchKMeans(**model_params["mini"])

    logging.info("Fitting labeled data...")
    clusterer.fit(labeled_data)

    # Step 3: Propagate labels to the rest of the data
    distances = clusterer.transform(unlabeled_data)
    closest_clusters = np.argmin(distances, axis=1)
    propagated_labels = np.empty_like(closest_clusters)

    for cluster in np.unique(closest_clusters):
        mask = closest_clusters == cluster
        if len(labeled_labels[clusterer.labels_ == cluster]) == 0: 
            continue
        most_common_label = np.bincount(labeled_labels[clusterer.labels_ == cluster]).argmax()
        propagated_labels[mask] = most_common_label

    # logging.info("Calculate Silhouette score...")
    # score = silhouette_score(data, labels)
    # logging.info("Silhouette score: " + str(score)) 

    all_labels = np.concatenate((labeled_labels, propagated_labels), axis=0)
    # Merge the labeled and unlabeled data
    data = np.concatenate((labeled_data, unlabeled_data), axis=0)
    return data, all_labels

def label_captured_data(prob_config: ProblemConfig, model_params = None):

    data = pd.read_parquet(prob_config.train_x_path)
    columns = data.columns
    labeled_data = data.to_numpy()
    labeled_labels = pd.read_parquet(prob_config.train_y_path).to_numpy().squeeze()
    ml_type = prob_config.ml_type
    # print(labeled_labels.squeeze().shape)

    logging.info("Load captured data")

    captured_x = pd.DataFrame()
    for file_path in tqdm(prob_config.captured_data_dir.glob("*.parquet"), ncols=100, desc ="Loading...", unit ="file"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])
    
    logging.info('Preprocessing captured data....')
    if prob_config.prob_id == 'prob-1':
        # path_save = "./src/model_config/phase-1/prob-1/sub_values.pkl"
        # extractor = FeatureExtractor(None, path_save)
        # unlabeled_data = extractor.load_new_feature(captured_x)
        # unlabeled_data = unlabeled_data[columns].to_numpy()
        unlabeled_data = captured_x[columns].to_numpy()
    else: 
        unlabeled_data = captured_x[columns].to_numpy()

    n_captured = len(unlabeled_data)
    n_samples = len(labeled_data) + n_captured

    logging.info(f"Loaded {n_captured} captured samples")

    print('unlabled: ', unlabeled_data.shape)
    print('labled: ', labeled_data.shape)

    logging.info("Initialize and fit the clustering model")

    
    data, approx_label = propagate_labels(labeled_data, labeled_labels, unlabeled_data)
    print(np.unique(approx_label))

    logging.info("Saving new data...")
    captured_x = pd.DataFrame(data, columns = columns)
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])
                              
    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)
    print(captured_x.info(), '\n', np.unique(approx_label_df, return_counts=True))
    logging.info(f"after process have {len(data)}  train + captured")
    logging.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)

    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    
    # label_captured_data(prob_config, model_params)
    label_captured_data(prob_config)

