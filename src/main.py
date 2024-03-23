import pandas as pd
import time
from datetime import datetime
from loguru import logger
from src.df_preprocessing import TextPreProcessing
from src.dict_preprocessing import DictPreProcessing
from src.sentiment_dict_approach import SentimentScoreDict


def load_files():
    input_data = pd.read_csv("./data/raw/zacks_arguments.csv", encoding="latin1")
    negative_words_df = pd.read_csv("./data/dictionaries/LM2018N.csv")
    positive_words_df = pd.read_csv("./data/dictionaries/LM2018P.csv")
    return input_data, positive_words_df, negative_words_df


def main():
    start_time = datetime.now()
    logger.info(f"Starting the pipeline at {start_time}")
    try:
        input_data, positive_words_df, negative_words_df = load_files()
        text_processor = TextPreProcessing(data=input_data, column="arguments_clean")
        dict_processor = DictPreProcessing(positive_words_df, negative_words_df)
        dict_sentiment = dict_processor.dict_processing_flow()
        input_data_processed = text_processor.preprocessing_flow()

        sentiment_calculator = SentimentScoreDict(
            data=input_data_processed,
            col="processed_arguments",
            dictionary=dict_sentiment,
        )

        input_data_w_sentiment = sentiment_calculator.sentiment_score_flow()

        logger.info(f"Processed data: {input_data_w_sentiment.head(10)}")

        input_data_w_sentiment.to_csv(
            "./data/processed/zacks_arguments_sentiment.csv", index=False
        )

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline finished in {duration.total_seconds()} seconds")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
