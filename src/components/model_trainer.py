'''


from dataclasses import dataclass
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Dict
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import argparse

# /c:/Users/abhisar/Downloads/ml_project/src/components/model_trainer.py
"""
Generic model trainer for typical sklearn pipelines.

- Loads transformed train/test artifacts (numpy, npz, csv, pkl, parquet, feather, or pandas DataFrame)
    and assumes target is the last column if single file per split.
- Automatically picks classification vs regression (or use config.task).
- Trains a RandomForest (fast reasonable default), evaluates, and saves the fitted model.
- Designed to integrate into small ML projects; adapt model/params as needed.
"""


from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        r2_score,
        mean_squared_error,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainerConfig:
        train_path: str  # path to transformed train artifact
        test_path: str   # path to transformed test artifact
        output_dir: str  # where to save model / reports
        model_name: str = "model.joblib"
        task: str = "auto"  # 'auto', 'classification', or 'regression'
        random_state: int = 42
        n_jobs: int = -1
        rf_kwargs: Optional[Dict] = None


class ModelTrainer:
        def __init__(self, config: ModelTrainerConfig):
                self.cfg = config
                self.model = None
                os.makedirs(self.cfg.output_dir, exist_ok=True)

        def _load_any(self, path: str) -> Any:
                ext = os.path.splitext(path)[1].lower()
                if ext in (".npy",):
                        return np.load(path)
                if ext in (".npz",):
                        return np.load(path)
                if ext in (".pkl", ".joblib"):
                        return joblib.load(path)
                if ext in (".csv",):
                        return pd.read_csv(path)
                if ext in (".parquet",):
                        return pd.read_parquet(path)
                if ext in (".feather",):
                        return pd.read_feather(path)
                # fallback: try joblib then numpy
                try:
                        return joblib.load(path)
                except Exception:
                        return np.load(path, allow_pickle=True)

        def _array_to_xy(self, obj: Any) -> Tuple[np.ndarray, np.ndarray]:
                # If pandas DataFrame
                if isinstance(obj, pd.DataFrame):
                        arr = obj.values
                else:
                        arr = np.asarray(obj)
                if arr.ndim == 1:
                        # 1D array of targets only -> no features
                        X = np.empty((arr.shape[0], 0))
                        y = arr
                        return X, y
                if arr.ndim == 2:
                        # Assume last column is target
                        X = arr[:, :-1]
                        y = arr[:, -1]
                        return X, y
                # For .npz with named arrays try common names
                if isinstance(obj, np.lib.npyio.NpzFile):
                        for key in ("arr_0", "X", "x", "X_train"):
                                if key in obj:
                                        X = obj[key]
                                        break
                        else:
                                raise ValueError("Cannot locate features inside npz")
                        for key in ("y", "y_train", "target"):
                                if key in obj:
                                        y = obj[key]
                                        break
                        else:
                                raise ValueError("Cannot locate target inside npz")
                        return np.asarray(X), np.asarray(y)
                raise ValueError("Unsupported data object shape/type for splitting into X/y")

        def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                logger.info("Loading train data from %s", self.cfg.train_path)
                train_obj = self._load_any(self.cfg.train_path)
                logger.info("Loading test data from %s", self.cfg.test_path)
                test_obj = self._load_any(self.cfg.test_path)

                X_train, y_train = self._array_to_xy(train_obj)
                X_test, y_test = self._array_to_xy(test_obj)

                # Ensure numeric arrays
                X_train = np.asarray(X_train)
                X_test = np.asarray(X_test)
                y_train = np.asarray(y_train).ravel()
                y_test = np.asarray(y_test).ravel()
                logger.info("Train shape X=%s y=%s", X_train.shape, y_train.shape)
                logger.info("Test shape  X=%s y=%s", X_test.shape, y_test.shape)
                return X_train, y_train, X_test, y_test

        def _guess_task(self, y: np.ndarray) -> str:
                if self.cfg.task in ("classification", "regression"):
                        return self.cfg.task
                # heuristic: integer or few unique values => classification
                if np.issubdtype(y.dtype, np.integer) or y.dtype == object:
                        if np.unique(y).shape[0] <= 50:
                                return "classification"
                # float with many unique -> regression
                if np.issubdtype(y.dtype, np.floating):
                        return "regression"
                # fallback
                return "classification"

        def _build_model(self, task: str):
                rf_kwargs = dict(random_state=self.cfg.random_state, n_jobs=self.cfg.n_jobs)
                if self.cfg.rf_kwargs:
                        rf_kwargs.update(self.cfg.rf_kwargs)
                if task == "classification":
                        return RandomForestClassifier(**rf_kwargs)
                return RandomForestRegressor(**rf_kwargs)

        def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray], task: str) -> Dict:
                metrics = {}
                if task == "classification":
                        try:
                                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                        except Exception as e:
                                logger.warning("Classification basic metrics failed: %s", e)
                        # try ROC AUC for binary or if probas shape matches
                        if y_proba is not None:
                                try:
                                        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                                                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                                        else:
                                                # multiclass ovR
                                                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                                except Exception:
                                        pass
                else:
                        metrics["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
                        metrics["r2"] = float(r2_score(y_true, y_pred))
                return metrics

        def train(self):
                X_train, y_train, X_test, y_test = self.load_data()
                task = self._guess_task(y_train)
                logger.info("Detected task: %s", task)
                self.model = self._build_model(task)
                logger.info("Fitting model...")
                self.model.fit(X_train, y_train)
                logger.info("Predicting on test set...")
                y_pred = self.model.predict(X_test)
                y_proba = None
                try:
                        if hasattr(self.model, "predict_proba"):
                                y_proba = self.model.predict_proba(X_test)
                except Exception:
                        y_proba = None
                metrics = self.evaluate(y_test, y_pred, y_proba, task)
                logger.info("Evaluation metrics: %s", metrics)
                # save model and simple report
                model_path = os.path.join(self.cfg.output_dir, self.cfg.model_name)
                joblib.dump(self.model, model_path)
                logger.info("Saved model to %s", model_path)
                report_path = os.path.join(self.cfg.output_dir, "train_report.pkl")
                joblib.dump({"metrics": metrics, "task": task}, report_path)
                logger.info("Saved report to %s", report_path)
                return {"model_path": model_path, "report_path": report_path, "metrics": metrics}


if __name__ == "__main__":
        # Example usage:
        # python model_trainer.py --train path/to/train.npy --test path/to/test.npy --out ./artifacts

        parser = argparse.ArgumentParser(description="Train model from transformed artifacts")
        parser.add_argument("--train", required=True, help="path to transformed train artifact")
        parser.add_argument("--test", required=True, help="path to transformed test artifact")
        parser.add_argument("--out", required=True, help="output directory to save model and report")
        parser.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
        args = parser.parse_args()

        cfg = ModelTrainerConfig(
                train_path=args.train,
                test_path=args.test,
                output_dir=args.out,
                task=args.task,
        )
        trainer = ModelTrainer(cfg)
        result = trainer.train()
        logger.info("Training finished. Result: %s", result)

'''

import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:          
            raise CustomException(e,sys)

