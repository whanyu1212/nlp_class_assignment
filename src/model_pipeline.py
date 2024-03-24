import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from lightgbm import LGBMClassifier
from src.lgb_tuner import LGBHyperparameterTuner


class ModelPipeline:
    def __init__(self, data: pd.DataFrame, text_column: str, label_column: str):

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        if not isinstance(text_column, str):
            raise ValueError(
                "The selected must be a string variable before vectorization"
            )

        if not isinstance(label_column, str):
            raise ValueError("The selected must be a string variable before encoding")

        self.data = data
        self.text_column = text_column
        self.label_column = label_column
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()

    def tfidf_vectorization(self, df: pd.DataFrame) -> np.ndarray:
        tfidf_matrix = self.vectorizer.fit_transform(df[self.text_column])
        return tfidf_matrix

    def label_encoding(self, df: pd.DataFrame) -> np.ndarray:
        self.label_encoder.fit(df[self.label_column])
        y = self.label_encoder.transform(df[self.label_column])
        return y

    def tune_model(self, X: np.ndarray, y: np.ndarray) -> float:
        tuner = LGBHyperparameterTuner(X, y)

        # Run the hyperparameter tuning
        best_params = tuner.create_optuna_study("lightgbm_model", "1")

        return best_params

    def create_model_with_best_params(self, X, y, best_params: dict) -> LGBMClassifier:
        """
        Initialize a model with the best hyperparameters found through
        optuna tuning.

        Args:
            best_params (dict): a dictionary of best hyperparameters (combination)

        Returns:
            LGBMClassifier: model
        """
        lgbm_cl = LGBMClassifier(**best_params)
        lgbm_cl.fit(X, y)
        with open("./models/lgbm_model.pkl", "wb") as f:
            pickle.dump(lgbm_cl, f)
        return lgbm_cl

    def evaluate_set(self, model: LGBMClassifier, X, y) -> dict:
        """
        Evaluate the model performance on a given set using a series of
        metrics.

        Args:
            model (LGBMClassifier): model fitted in the previous step
            data (pd.DataFrame): data set to evaluate the model on

        Returns:
            dict: a dictionary of metrics and scores
        """
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y, y_pred_proba, multi_class="ovr"),
        }

        return metrics

    # def generate_confusion_matrix(
    #     self, model: LGBMClassifier, test: pd.DataFrame
    # ) -> np.ndarray:
    #     """
    #     Generate the confusion matrix for the test set.

    #     Args:
    #         model (LGBMClassifier): model fitted in the previous step
    #         test (pd.DataFrame): test set that the model has not seen before

    #     Returns:
    #         np.ndarray: confusion matrix
    #     """
    #     y_pred = model.predict(self.X)

    #     cm = confusion_matrix(self.X, y_pred, labels=model.classes_)
    #     disp = ConfusionMatrixDisplay(
    #         confusion_matrix=cm, display_labels=model.classes_
    #     )
    #     disp.plot(cmap="Blues")
    #     plt.xlabel("Predicted", fontsize=14)
    #     plt.ylabel("Truth", fontsize=14)
    #     plt.title("Confusion Matrix", fontsize=16)
    #     plt.savefig("./output/confusion_matrix.png", bbox_inches="tight")

    #     return cm

    def modelling_flow(self):
        X = self.tfidf_vectorization(self.data)
        y = self.label_encoding(self.data)
        best_params = self.tune_model(X, y)
        model = self.create_model_with_best_params(X, y, best_params)
        metrics = self.evaluate_set(model, X, y)
        print(metrics)
        return model, metrics
