import pandas as pd
import spacy


class DictPreProcessing:
    def __init__(self, p_df, n_df):
        self.p_words = p_df.iloc[:, 0].tolist()
        self.n_words = n_df.iloc[:, 0].tolist()
        self.nlp = spacy.load("en_core_web_sm")

    def lower_case(self, lst):
        return [word.lower() for word in lst]

    def lemmatize_words(self, lst):
        doc = self.nlp(" ".join(lst))
        lemmatized_p_words = set([token.lemma_ for token in doc])
        return list(lemmatized_p_words)

    def create_dict(self, p_words, n_words):
        dict_ = {}
        dict_["positive"] = p_words
        dict_["negative"] = n_words
        return dict_

    def dict_processing_flow(self):
        p_words = self.p_words.copy()
        n_words = self.n_words.copy()
        p_words = self.lower_case(p_words)
        n_words = self.lower_case(n_words)
        p_words = self.lemmatize_words(p_words)
        n_words = self.lemmatize_words(n_words)
        dict_ = self.create_dict(p_words, n_words)
        return dict_
