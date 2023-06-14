import argparse
import logging

import mlflow
import numpy as np
import xgboost as xgb
import catboost 
from collections import Counter
from utils import *
import json
import yaml
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

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

        class_model = Models(prob_config)
        class_model.xgb_classifier()

        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{class_model.EXPERIMENT_NAME}"
        )

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        train_x = train_x.to_numpy()
        train_y = train_y.to_numpy()
        logging.info(f"loaded {len(train_x)} samples")

        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()
            train_x = np.concatenate((train_x, captured_x))
            train_y = np.concatenate((train_y, captured_y))
            logging.info(f"added {len(captured_x)} captured samples")

        val = int(len(train_x)-len(train_x)*0.125)

        # counter = Counter(train_y)
        # # estimate scale_pos_weight value
        # print(counter[1] / counter[0])
        # return

        
        print(f'Train data samples: {val}, val data samples" {len(train_x)-val}')


        model = class_model.model
        print(class_weight.compute_class_weight(class_weight = 'balanced',
                                                                classes = np.unique(train_y),
                                                                y = train_y))
        model.fit(train_x[:val], train_y[:val], eval_set=[(train_x[val:], train_y[val:])],
                  **class_model.train)



        # evaluate
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        predictions = model.predict(test_x)
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
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--config-path", type=str, default="./src/model_config/xgboost.yaml")
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    
    # if os.path.exists(prob_config.captured_x_path):
    #     args.add_captured_data = True
    ModelTrainer.train_model(
        prob_config, add_captured_data=args.add_captured_data
    )
