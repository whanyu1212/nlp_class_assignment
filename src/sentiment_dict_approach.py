import pandas as pd
from fuzzywuzzy import process
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from src.utils import parse_cfg

tqdm.pandas()


class SentimentScoreDict:
    def __init__(
        self,
        data: pd.DataFrame,
        col: str,
        dictionary: dict,
        threshold: int = 90,
        max_length_diff: int = 3,
    ):
        self.data = data
        self.col = col
        self.dictionary = dictionary
        self.threshold = threshold
        self.max_length_diff = max_length_diff
        self.negation_words = parse_cfg("./config/parameters.yaml")["negation_words"]

    def find_matches_without_negate(self, words, sentiment):
        matches = []
        for word in words:
            for pos_word in self.dictionary[sentiment]:
                # Skip if the length difference is beyond the allowed threshold
                if abs(len(word) - len(pos_word)) > self.max_length_diff:
                    continue
                best_match, score = process.extractOne(word, [pos_word])
                if score >= self.threshold:
                    matches.append((word, best_match, score))
                    break  # Assuming only one match is needed
        return len(matches)

    def find_matches_with_negate(self, words, sentiment):
        matches = []
        skip_next = False
        for i, word in enumerate(words):
            # Check if the word is a negation and mark to skip or invert next word's sentiment
            if word in self.negation_words:
                skip_next = True
                continue  # Move to the next word in the list

            # If this word should be skipped because it follows a negation, reset skip_next and continue
            if skip_next:
                skip_next = False  # Reset for the next iteration
                continue  # Skip this word

            for pos_word in self.dictionary[sentiment]:
                # Skip if the length difference is beyond the allowed threshold
                if abs(len(word) - len(pos_word)) > self.max_length_diff:
                    continue
                best_match, score = process.extractOne(word, [pos_word])
                if score >= self.threshold:
                    matches.append((word, best_match, score))
                    break  # Found a match, no need to check other sentiment words

        return len(matches)

    def calculate_sentiment_score_without_negate(self, data: pd.DataFrame):
        # loop through the text and calculate the sentiment score
        data["p_matches_without_negate"] = data[self.col].progress_apply(
            self.find_matches_without_negate, sentiment="positive"
        )
        data["n_matches_without_negate"] = data[self.col].progress_apply(
            self.find_matches_without_negate, sentiment="negative"
        )

        data["score_without_negate"] = (
            data["p_matches_without_negate"] - data["n_matches_without_negate"]
        ) / (data["p_matches_without_negate"] + data["n_matches_without_negate"] + 1)

        return data

    def calculate_sentiment_score_with_negate(self, data: pd.DataFrame):
        # loop through the text and calculate the sentiment score
        data["p_matches_with_negate"] = data[self.col].progress_apply(
            self.find_matches_with_negate, sentiment="positive"
        )
        data["n_matches_with_negate"] = data[self.col].progress_apply(
            self.find_matches_with_negate, sentiment="negative"
        )

        data["score_with_negate"] = (
            data["p_matches_with_negate"] - data["n_matches_with_negate"]
        ) / (data["p_matches_with_negate"] + data["n_matches_with_negate"] + 1)

        return data

    def get_compound_score(self, text):
        sia = SentimentIntensityAnalyzer()
        text_str = " ".join(text)
        return sia.polarity_scores(text_str)["compound"]

    def sentiment_score_flow(self):
        data = self.data.copy()
        data = self.calculate_sentiment_score_without_negate(data)
        data = self.calculate_sentiment_score_with_negate(data)
        data["compound_score"] = data[self.col].progress_apply(self.get_compound_score)
        return data
