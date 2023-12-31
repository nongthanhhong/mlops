

import json
import mlflow
import logging
import catboost 
import argparse
import numpy as np
from utils import *
import catboost as cb
import pandas as pd
import xgboost as xgb
from collections import Counter
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig

def load(**kwargs):
    config = {}
    for k, v in kwargs.items():
      if type(v) == dict:
        v = load(**v)
      config[k] = v
    return config

class Models:
    def __init__(self, prob_config):
        self.EXPERIMENT_NAME = None
        self.config_folder = "./src/model_config"
        self.phase = prob_config.phase_id
        self.prob = prob_config.prob_id
        self.model = None
        self.params = None
        self.train = None
    
    def read_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        return load(**config)

    def xgb_classifier(self):
         
        config =  self.read_config(self.config_folder+ "/" + self.phase + "/" + self.prob + "/xgb.json")
        print(config)
        self.EXPERIMENT_NAME = config["meta_data"]["model_name"]
        self.params = config["params"]
        self.train = config["train"]
        self.model = xgb.XGBClassifier(**self.params)

    def catboost_classifier(self):
        
        config =  self.read_config(self.config_folder+ "/" + self.phase + "/" + self.prob + "/catboost.json")
        self.EXPERIMENT_NAME = config["meta_data"]["model_name"]
        self.params = config["params"]
        self.train = config["train"]
        self.model = catboost.CatBoostClassifier(**self.params)

class ModelTrainer:

    @staticmethod
    def train_model(prob_config: ProblemConfig, add_captured_data=False):

        #define model
        class_model = Models(prob_config)
        class_model.catboost_classifier()

        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{class_model.EXPERIMENT_NAME}"
        )


        # load train data
        if add_captured_data:
            logging.info("Use captured data")
            
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)

            # Merge the labeled and unlabeled data
            all_data = pd.concat([train_x, test_x, captured_x], axis=0)
            all_labels = pd.concat([train_y, test_y, captured_y], axis=0)
           
            weight = int( len(train_y) / len(captured_y) ) if len(train_y)>len(captured_y) else 1
            weights = np.concatenate([np.ones(len(train_y)), np.ones(len(test_y)), np.ones(len(captured_y)) * weight])

            # split data into training, validation, and test sets
            train_x, test_x, train_y, test_y, train_weights, test_weights = train_test_split(all_data, all_labels, weights, 
                                                                                             test_size=0.2, 
                                                                                             random_state=42,
                                                                                             stratify= all_labels)
            train_x, val_x, train_y, val_y, train_weights, val_weights = train_test_split(train_x, train_y, train_weights, 
                                                                                          test_size=0.25, 
                                                                                          random_state=42,
                                                                                          stratify= train_y)

            print('Train: old - new: ', np.unique(train_weights, return_counts=True))
            print('Val: old - new: ', np.unique(val_weights, return_counts=True))
            print('Test: old - new: ', np.unique(test_weights, return_counts=True))

            # return

            # create Pool objects for each set with weights
            dtrain = cb.Pool(train_x, label=train_y, weight=train_weights)
            dval = cb.Pool(val_x, label=val_y, weight=val_weights)
            dtest = cb.Pool(test_x, label=test_y, weight=test_weights)

        else:
            logging.info("Use original data")
            train_x, train_y = RawDataProcessor.load_train_data(prob_config)
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            test_x, test_y = RawDataProcessor.load_test_data(prob_config)

            train_x, val_x, train_y, val_y = train_test_split(
                                                    train_x, train_y,
                                                    test_size=0.125,
                                                    random_state=42,
                                                    stratify= train_y)
            
            dtrain = cb.Pool(train_x, label=train_y)
            dval =  cb.Pool(val_x, label=val_y)
            dtest =  cb.Pool(test_x, label=test_y)
        

        counter = Counter(train_y)
        # estimate scale_pos_weight value
        print(f'num 1: {counter[1]} - {100*counter[1]/len(train_y)}%, num 0 {counter[0]} - {100*counter[0]/len(train_y)}%')
        

        print(f'Loaded {len(train_y)} Train samples, {len(val_y)} val samples , and {len(test_y)} test samples!')


        


        model = class_model.model

        model.fit(dtrain, 
                  eval_set=dval,
                  **class_model.train)
        
        # evaluate
        predictions = model.predict(dtest)

        counter = Counter(test_y)
        # estimate scale_pos_weight value
        print(f'num 1: {counter[1]} - {100*counter[1]/len(test_y)}%, num 0 {counter[0]} - {100*counter[0]/len(test_y)}%')
        
        auc_score = roc_auc_score(test_y, predictions)

        metrics = {"test_auc": auc_score}
        logging.info(f"metrics: {metrics}")
        logging.info("\n" + classification_report(test_y, predictions))
        logging.info("\n" + str(confusion_matrix(test_y, predictions)))

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )
        mlflow.end_run()

        # Plot the ROC curve
        fpr, tpr, _ = roc_curve(test_y, predictions)
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)

    ModelTrainer.train_model(
        prob_config, add_captured_data=args.add_captured_data
    )
