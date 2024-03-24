import json
import pickle
import time
from typing import Any, Dict

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMModel
from mlflow.exceptions import MlflowException
from optuna.samplers import TPESampler
from optuna.trial import Trial
from sklearn.model_selection import cross_val_score


class LGBHyperparameterTuner:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the HyperparameterTuner class with the given
        parameters. The train and val sets are already splitted using
        the time series splitting method. There is no point doing CV
        since its not applicable in this context.

        Args:
            X_train (pd.DataFrame): Features of the training data
            y_train (pd.Series): Response of the training data
            X_val (pd.DataFrame): Features of the validation data
            y_val (pd.Series): Response of the validation data
        """

        self.X = X
        self.y = y

    def create_or_get_experiment(self, name: str) -> str:
        """
        Create or get an mlflow experiment based on the experiment name
        specified.

        Args:
            name (str): name to be given to the experiment
            or name of the experiment to be retrieved

        Raises:
            ValueError: if the experiment name is not found

        Returns:
            str: experiment ID in string format
        """
        try:
            experiment_id = mlflow.create_experiment(name)
        except MlflowException:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                raise ValueError("Experiment not found.")
        return experiment_id

    def log_model_and_params(
        self, model: LGBMModel, trial: Trial, params: Dict[str, Any], cv_accuracy: float
    ):
        """
        Log the model, params, and mean accuracy from mlflow
        experiments.

        Args:
            model (LGBMModel): the lightgbm trained every trial
            trial (Trial): the optuna trial
            params (Dict[str, Any]): the parameters used for the lightgbm model
            pr_auc (float): the PR AUC of each trial
        """
        # logs the model, params, and pr_auc of a trial
        mlflow.lightgbm.log_model(model, "lightgbm_model")
        mlflow.log_params(params)
        mlflow.log_metric("Mean Accuracy", cv_accuracy)
        # storing a pickled version of the best model
        trial.set_user_attr(key="best_booster", value=pickle.dumps(model))

    def objective(self, trial: Trial) -> float:
        """
        Define the objective function for the optuna study. Start the
        mlflow run and log the model, params, and pr_auc of a trial.

        Args:
            trial (Trial): a specific trial of the optuna study

        Returns:
            float: score of the PR AUC of the model during the trial
        """

        experiment_id = self.create_or_get_experiment("lightgbm-optuna")
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10.0),
                "num_leaves": trial.suggest_int("num_leaves", 2, 64),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.8),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_depth": trial.suggest_int("max_depth", -1, 10),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 200),
            }

            lgbm_cl = LGBMClassifier(**params)

            lgbm_cl.fit(self.X, self.y)
            cv_accuracy = cross_val_score(
                lgbm_cl, self.X, self.y, cv=5, scoring="accuracy"
            ).mean()

            self.log_model_and_params(lgbm_cl, trial, params, cv_accuracy)

        return cv_accuracy

    def create_optuna_study(
        self,
        model_name: str,
        model_version: str,
        n_trials: int = 1,
        max_retries: int = 3,
        delay: int = 5,
    ) -> dict:
        """
        Create and orchestrate an optuna study to optimize the
        hyperparameters of the lightgbm model. Retry mechanism is added
        to mitigate transient errors.

        Args:
            model_name (str): model name assigned to the model
            model_version (str): model version assigned to the model
            n_trials (int, optional): The number of trials. Defaults to 20.
            max_retries (int, optional): Max number of retries if exception occurs.
            Defaults to 3.
            delay (int, optional): Time out before retrying. Defaults to 5.

        Raises:
            RuntimeError: If the study fails to optimize after maximum retries

        Returns:
            dict: the best parameters from the study
        """

        study = optuna.create_study(
            study_name="optimizing lightgbm",
            direction="maximize",
            sampler=TPESampler(seed=42),
        )
        best_params = None

        for _ in range(max_retries):
            try:
                study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)
                best_trial = study.best_trial
                best_params = best_trial.params
                break
            except Exception as e:
                print(f"An error occurred: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        else:
            raise RuntimeError("Failed to optimize the study after maximum retries")

        # save the combination of best hyperparameters to a json file
        with open("./output/best_param.json", "w") as outfile:
            json.dump(best_params, outfile)

        return best_params
