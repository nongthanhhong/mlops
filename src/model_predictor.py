import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from utils import *
from pandas.util import hash_pandas_object
from pydantic import BaseModel
import numpy as np
from gensim.models import KeyedVectors
import catboost as cb

from problem_config import ProblemConst, create_prob_config, load_feature_configs_dict
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


def input_process(feature_df, embedding_model):

    from_acc = feature_df['feature4'].astype(float).tolist()
    to_acc = feature_df['feature7'].astype(float).tolist()
    amount = feature_df['feature3'].tolist()
    feature_vector = []
    for i in range(len(from_acc)):
        if str(from_acc[i]) in embedding_model:
            from_embedding = embedding_model.get_vector(str(from_acc[i]))
        else: 
            from_embedding = embedding_model.most_similar(str(from_acc[i]))

        if str(to_acc[i]) in embedding_model:
            to_embedding = embedding_model.get_vector(str(to_acc[i]))
        else: 
            to_embedding = embedding_model.most_similar(str(to_acc[i]))

        # Calculate feature vector as concatenation of embeddings and transaction amount
        feature_vector.append(np.concatenate([from_embedding, to_embedding, [amount[i]]]))

    feature_df = feature_df.copy().assign(node_embedding = feature_vector)

    x_vec = np.array(feature_df["node_embedding"].tolist())
    x_other = feature_df.drop(columns=["node_embedding"])
    feature_df = pd.DataFrame(np.concatenate((x_other, x_vec), axis=1))
    logging.info(feature_df)

    return feature_df 


class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        self.model = mlflow.pyfunc.load_model(model_uri)

        self.columns_to_keep = self.prob_config.categorical_cols + self.prob_config.numerical_cols

        #using embedding model
        self.embedding_model = KeyedVectors.load('./src/model_config/phase-1/prob-1/node_embeddings.bin')

    def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        return random.choice([0, 1])

    def predict(self, data: Data):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )

        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )
        feature_df = feature_df[self.columns_to_keep]

        if self.prob_config.prob_id == 'prob-1':
            feature_df = input_process(feature_df, self.embedding_model)

        

        prediction = self.model.predict(feature_df)
        is_drifted = self.detect_drift(feature_df)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor_1: ModelPredictor, predictor_2: ModelPredictor):
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-1/prob-1/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor_1.predict(data)
            self._log_response(response)
            return response

        @self.app.post("/phase-1/prob-2/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor_2.predict(data)
            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    prob_1_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()

    prob_2_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE
        / ProblemConst.PROB2
        / "model-1.yaml"
    ).as_posix()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", nargs="+", default=[prob_1_config_path, prob_2_config_path])
    parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
    args = parser.parse_args()

    predictor_1 = ModelPredictor(config_file_path=args.config_path[0])
    predictor_2 = ModelPredictor(config_file_path=args.config_path[1])

    api = PredictorApi(predictor_1, predictor_2)
    api.run(port=args.port)
