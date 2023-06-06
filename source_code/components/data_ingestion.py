import os
import sys
import numpy as np

from source_code.utils import get_df_from_mongo, drop_missing_val_columns
from source_code.entity.artifact_entity import DataIngestionArtifact
from source_code.entity.config_entity import DataIngestionConfig
from source_code.exception import CustomException
from source_code.logger import logging
from source_code import utils


class DataIngestion:
    def __init__(self):
        logging.info(f"{'--' * 20} Data Ingestion {'--' * 20}")
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        This function initiates the data ingestion process
        :return:
        """
        logging.info("Initiating Data-Ingestion")
        try:
            df = get_df_from_mongo('mushrooms_data', 'mushroom')
            logging.info('Successfully read Mushrooms Dataset')

            logging.info(f"stalk-root: {df['stalk-root'].unique()}")
            logging.info("Replacing '?' in 'stalk-root' with np.nan")
            df = df.replace(to_replace='?', value=np.nan)
            logging.info(f"stalk-root: {df['stalk-root'].unique()}")

            # Dropping columns with more than a specified percent of missing values.
            threshold = self.data_ingestion_config.MISSING_VALUE_THRESHOLD
            logging.info(f'Checking for columns with more than {threshold}% missing values')
            missing_val_col_list = drop_missing_val_columns(df)
            logging.info(f'Columns found: {missing_val_col_list}')
            logging.warn(f'Dropping columns found with more than {threshold}% missing values')
            df = df.drop(missing_val_col_list, axis=1)

            # Dropping columns which have only one unique value as it will not help in prediction.
            logging.info('Checking for columns with single unique value')
            single_unique_val_col_list = utils.drop_single_unique_val_columns(df)
            logging.info(f'Columns found: {single_unique_val_col_list}')
            logging.warn('Dropping columns found with single unique value')
            df.drop(single_unique_val_col_list, axis=1, inplace=True)

            # Making directory to store all paths. Test & raw_data paths are also stored in same dir
            os.makedirs(os.path.dirname(self.data_ingestion_config.base_data_path), exist_ok=True)

            logging.info('Saving base_data_file to artifacts folder')
            df.to_csv(self.data_ingestion_config.base_data_path, index=False, header=True)

            logging.info('Preparing Artifacts')
            data_ingestion_artifact = DataIngestionArtifact(base_data_path=self.data_ingestion_config.base_data_path)

            logging.info("Data-Ingestion completed.")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
