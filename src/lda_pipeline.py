import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV


class LDAPipeline:
    def __init__(
        self, data: pd.DataFrame, text_column: str, n_topics: int, n_top_words: int = 10
    ):
        self.data = data
        self.text_column = text_column
        self.n_topics = n_topics
        self.n_top_words = n_top_words

    def create_bag_of_words(self):
        self.vectorizer = CountVectorizer()
        BoW_matrix = self.vectorizer.fit_transform(self.data[self.text_column])
        return BoW_matrix.toarray()

    def save_feature_names(self):
        feature_names = self.vectorizer.get_feature_names_out()
        return feature_names

    def tune_lda_model(self, BoW_matrix):
        param_grid = {
            "n_components": [2, 3, 4, 5],  # Number of topics
            "learning_decay": [0.5, 0.7, 0.9],  # Learning rate decay
            # Add other parameters here as needed
        }
        lda = LatentDirichletAllocation(random_state=0)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=lda, param_grid=param_grid, n_jobs=-1, cv=5
        )
        grid_search.fit(BoW_matrix)

        return grid_search.best_params_

    def fit_lda_model(self, BoW_matrix, best_params):
        lda = LatentDirichletAllocation(
            n_components=best_params["n_components"],
            learning_decay=best_params["learning_decay"],
            random_state=0,
        )
        lda.fit(BoW_matrix)
        return lda

    def display_topics(self, model, feature_names):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(
                " ".join(
                    [
                        feature_names[i]
                        for i in topic.argsort()[: -self.n_top_words - 1 : -1]
                    ]
                )
            )

    def lda_workflow(self):
        BoW_matrix = self.create_bag_of_words()
        feature_names = self.save_feature_names()
        best_params = self.tune_lda_model(BoW_matrix)
        lda_model = self.fit_lda_model(BoW_matrix, best_params)
        self.display_topics(lda_model, feature_names)
