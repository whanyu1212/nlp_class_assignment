import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()


class TextPreProcessing:
    def __init__(self, data: pd.DataFrame, column: str):
        """Initializes the class with the data and the column

        Args:
            data (pd.DataFrame): input dataframe
            column (str): the column that contains the text data

        Raises:
            ValueError: raise an error if the data is not a pandas DataFrame
            ValueError: raise an error if the data has no string columns
            ValueError: raise an error if the selected column is not a string
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if not any(data.dtypes == "object"):
            raise ValueError("data must contain at least one column of type string")

        if not isinstance(column, str):
            raise ValueError("The selected must be a string variable")

        self.data = data
        self.column = column
        self.nlp = spacy.load("en_core_web_sm")

    def lower_case(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all the alphabets in the string of text
        to lower case

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: dataframe with lower case text
            in the column that was selected
        """
        df[self.column] = df[self.column].str.lower()
        return df

    def remove_special_characters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove everything that is not alphanumeric or spaces, tabs, line
        breaks from the text.

        \w: This matches any word character (equal to [a-zA-Z0-9_])
        \s: This matches any whitespace character (spaces, tabs, line breaks)
        [^...]: The caret ^ inside the square brackets negates the set,
        meaning it matches any character not in the set

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with special characters removed
        """
        df[self.column] = df[self.column].str.replace("[^\w\s]", "", regex=True)
        return df

    def strip_extra_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """remove extra spaces from the text

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe with extra spaces removed from the
            text in the selected column
        """
        df[self.column] = df[self.column].str.strip().str.replace(" +", " ", regex=True)
        return df

    def apply_spacy_pipeline(self, text: str) -> list:
        """Apply the default spacy processing
        pipeline to the text

        Args:
            text (str): the string value in each
            row of the selected column

        Returns:
            list: a list of lemmatized tokens
        """
        doc = self.nlp(text)
        lemmatized_text = [token.lemma_ for token in doc if not token.is_stop]
        return lemmatized_text

    def preprocessing_flow(self):
        df = self.data.copy()
        df = self.lower_case(df)
        df = self.remove_special_characters(df)
        df = self.strip_extra_spaces(df)
        # its easier to use list for dictionary approach
        df[f"Processed_{self.column}_list"] = df[self.column].progress_apply(
            self.apply_spacy_pipeline
        )
        # its easier to use string for count vectorizer and tfidf
        df[f"Processed_{self.column}_str"] = df[f"Processed_{self.column}_list"].apply(
            " ".join
        )
        return df
