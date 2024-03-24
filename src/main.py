import pandas as pd
import time
from datetime import datetime
from loguru import logger
from src.df_preprocessing import TextPreProcessing
from src.dict_preprocessing import DictPreProcessing
from src.sentiment_dict_approach import SentimentScoreDict
from src.lgbm_pipeline import ModelPipeline


def load_files():
    zack_data = pd.read_csv("./data/raw/zacks_arguments.csv", encoding="latin1")
    negative_words_df = pd.read_csv("./data/dictionaries/LM2018N.csv")
    positive_words_df = pd.read_csv("./data/dictionaries/LM2018P.csv")
    startup_data = pd.read_excel(
        "/Users/hanyuwu/Study/nlp_class_assignment/data/raw/startups.xlsx",
        sheet_name="Request 2",
    )
    return zack_data, positive_words_df, negative_words_df, startup_data


def main():
    start_time = datetime.now()
    logger.info(f"Starting the pipeline at {start_time}")
    try:
        zack_data, positive_words_df, negative_words_df, startup_data = load_files()

        argument_text_processor = TextPreProcessing(
            data=zack_data, column="arguments_clean"
        )

        dict_processor = DictPreProcessing(positive_words_df, negative_words_df)
        processed_sentiment_dictionary = dict_processor.dict_processing_flow()
        zack_arguments_processed = argument_text_processor.preprocessing_flow()
        sentiment_calculator = SentimentScoreDict(
            data=zack_arguments_processed,
            col="Processed_arguments_clean_list",
            dictionary=processed_sentiment_dictionary,
        )
        zack_arguments_w_sentiment = sentiment_calculator.sentiment_score_flow()

        zack_arguments_w_sentiment.to_csv(
            "./data/processed/zacks_arguments_sentiment.csv", index=False
        )

        startup_text_processor = TextPreProcessing(
            data=startup_data, column="Description"
        )
        startup_data_processed = startup_text_processor.preprocessing_flow()

        startup_data_processed.to_csv(
            "./data/processed/startup_data_processed.csv", index=False
        )

        model_pipeline = ModelPipeline(
            data=startup_data_processed,
            text_column="Processed_Description_str",
            label_column="Industry",
        )

        model_pipeline.modelling_flow()

        # logger.info(
        #     f"Processed zack's arguments: {zack_arguments_w_sentiment.head(10)}"
        # )

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline finished in {duration.total_seconds()} seconds")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
