
import yaml
import json
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from utils import *
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
from data_engineering import DataAnalyzer, FeatureExtractor
from raw_data_processor import *


class ClusteringEvaluator:
    def __init__(self, X, y, model, n_cluster):
        self.X = X
        self.y = y
        self.model = model
        self.n_cluster = n_cluster

    def evaluate_clustering(self):

        # Calculate Silhouette Coefficient
        start_time = time.time()
        silhouette = metrics.silhouette_score(self.X, self.model.labels_)
        end_time = time.time()
        silhouette_time = end_time - start_time

        # Calculate Rand Index

        new_labels = []

        kmeans_clusters = self.model.predict(self.X)

        for cluster in range(self.n_cluster):
            mask = self.model.labels_ == cluster
            if len(self.y[mask]) == 0:
                # If no data points in the cluster, assign a default label (e.g., 0)
                new_labels.append(1)
                continue
            most_common_label = np.bincount(self.y[mask]).argmax()
            new_labels.append(most_common_label)

        approx_label = [new_labels[c] for c in kmeans_clusters]
        
        
        print('Truth labels : ', np.unique(self.y, return_counts= True))
        print('Predicted labels: ', np.unique(approx_label, return_counts=True))

        # compute contingency matrix (also called confusion matrix)
        start_time = time.time()

        contingency_matrix = metrics.cluster.contingency_matrix(self.y, approx_label)
        # calculate purity for each cluster
        purity = np.amax(contingency_matrix, axis=0) / np.sum(contingency_matrix, axis=0)
        end_time = time.time()
        purity_time = end_time - start_time



        start_time = time.time()
        rand_index = metrics.adjusted_rand_score(self.y, approx_label)
        end_time = time.time()
        rand_index_time = end_time - start_time

        # Calculate Sum of Squared Distance (SSD)
        start_time = time.time()
        ssd = self.model.inertia_
        end_time = time.time()
        ssd_time = end_time - start_time

        logging.info(f'Silhouette Coefficient: {silhouette:.2f} (Elapsed time: {silhouette_time:.2f} seconds)')
        logging.info(f'Rand Index: {rand_index:.2f} (Elapsed time: {rand_index_time:.2f} seconds)')
        logging.info(f'Sum of Squared Distance (SSD): {ssd:.2f} (Elapsed time: {ssd_time:.2f} seconds)')
        logging.info(f'Avg Purity of clustering: {np.mean(purity):.2f} (Elapsed time: {purity_time:.2f} seconds)')


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
        n_cluster = model_params["k_means"]["n_clusters"]

    elif algorithm == 'MiniBatchKMeans':
        # Step 2: Cluster the data using MiniBatchKMeans
        logging.info(f"Parameters: {model_params['mini']}")
        clusterer = MiniBatchKMeans(**model_params["mini"])
        n_cluster = model_params["mini"]["n_clusters"]

    logging.info("Fitting labeled data...")
    
    start_time = time.time()
    clusterer.fit(labeled_data)
    end_time = time.time()
    logging.info(f'Elapsed time: {(end_time - start_time):.2f}')

    logging.info('Evaluate cluster model... ')
    evaluator = ClusteringEvaluator(labeled_data, labeled_labels, clusterer, n_cluster)
    evaluator.evaluate_clustering()

    # Step 3: Propagate labels to the rest of the data
    logging.info("Labeling new data...")
    new_labels = []

    kmeans_clusters = clusterer.predict(unlabeled_data)
    
    for cluster in range(n_cluster):
        mask = clusterer.labels_ == cluster
        if len(labeled_labels[mask]) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(1)
            continue
        most_common_label = np.bincount(labeled_labels[mask]).argmax()
        new_labels.append(most_common_label)

    propagated_labels = [new_labels[c] for c in kmeans_clusters]

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
    captured_x = captured_x.drop_duplicates()

    
    logging.info('Preprocessing captured data....')
    if prob_config.prob_id == 'prob-1':
        path_save = "./src/model_config/phase-1/prob-1/sub_values_captured.pkl"
        os.remove(path_save)
        extractor = FeatureExtractor(captured_x, path_save)
        unlabeled_data = extractor.create_new_feature(captured_x)
        unlabeled_data = unlabeled_data[columns].to_numpy()

    else: 
        unlabeled_data = captured_x[columns].to_numpy()

    n_captured = len(unlabeled_data)
    n_samples = len(labeled_data) + n_captured

    logging.info(f"Loaded {n_captured} captured samples")

    print('unlabled: ', unlabeled_data.shape)
    print('labled: ', labeled_data.shape)

    logging.info("Initialize and fit the clustering model")


    data, approx_label = propagate_labels(labeled_data, labeled_labels, unlabeled_data)
    # print(np.unique(approx_label))

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

