
import os
import yaml
import time
import mlflow
import random
import logging
import uvicorn
import argparse
import numpy as np
from utils import *
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from data_engineering import FeatureExtractor, DataAnalyzer

from problem_config import ProblemConst, create_prob_config, load_feature_configs_dict
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):

        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        # mlflow.pyfunc.get_model_dependencies(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join("models:/", self.config["model_name"], str(self.config["model_version"]))
        self.model = mlflow.pyfunc.load_model(model_uri)

        self.columns_to_keep = self.prob_config.categorical_cols + self.prob_config.numerical_cols

        path_save = "./src/model_config/phase-1/prob-1/sub_values.pkl"

        if os.path.isfile("./src/model_config/phase-1/prob-1/sub_values_captured.pkl"):
            path_save = "./src/model_config/phase-1/prob-1/sub_values_captured.pkl"

        self.extractor = FeatureExtractor(None, path_save)
        
        self.eda =  DataAnalyzer(self.prob_config)

    def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        return random.choice([0, 1])

    def predict(self, data: Data):

        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)

        if self.prob_config.prob_id == 'prob-1':
            
            feature_df = RawDataProcessor.apply_category_features(
                                            raw_df=raw_df,
                                                 categorical_cols=self.prob_config.categorical_cols,
                                                        category_index=self.category_index, 
                                                            raw_config = self.prob_config.raw_feature_config_path)

            new_feature = self.extractor.create_new_feature(feature_df)
            new_feature_df = new_feature[self.columns_to_keep]

        else:
            
            feature_df = RawDataProcessor.apply_category_features(
                                            raw_df=raw_df,
                                                categorical_cols=self.prob_config.categorical_cols,
                                                    category_index=self.category_index)

            new_feature_df = feature_df[self.columns_to_keep]

        new_feature_df = self.eda.preprocess_data(input_data=new_feature_df)

        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )
        logging.info(f'Size of input: {len(new_feature_df)}')
        
        # logging.info(new_feature_df)
        # prediction = self.model.predict(feature_df)
        
        prediction = self.model.predict(new_feature_df)
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
