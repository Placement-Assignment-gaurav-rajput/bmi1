import sys
from typing import Optional, Any

import numpy as np
import pandas as pd
# from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from source_code.logger import logging
from source_code.exception import CustomException
from source_code.entity import config_entity, artifact_entity
from source_code.utils import save_numpy_array_data, save_object, cramers_V
from source_code.entity.config_entity import TARGET_COLUMN, DataTransformationConfig


class DataTransformation:
    """
    DataTransformation takes input as the artifacts of Data Ingestion process, pre-processes the data; saves the
    training and testing numpy arrays in artifacts folder and returns the path of saved locations.
    """
    def __init__(self, data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        logging.info(f"{'--' * 20} Data Transformation {'--' * 20}")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformation_object(column_list: list):
        """
        This function pre-processes the given dataset
        :param column_list: A list of names(str format) of input features
        :return: An object of ColumnTransformer object which pre-process input features
        """
        try:
            input_feature_pipe = Pipeline(steps=[("Imputer", SimpleImputer(strategy="most_frequent")),
                                                 ("OHE", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                                                 ("std_scaler", StandardScaler(with_mean=False))])

            transformer = ColumnTransformer([('pipe_1', input_feature_pipe, column_list)], remainder='drop')
            return transformer

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        """
        This function initiates Data Transformation process
        :return: paths of training numpy array, test numpy array and pre-processing object
        """
        logging.info('Data Transformation initiated')
        try:
            logging.info('Reading Base Data')
            base_df = pd.read_csv(self.data_ingestion_artifact.base_data_path)

            logging.info("Encoding the TARGET feature")
            base_df[TARGET_COLUMN] = base_df[TARGET_COLUMN].map({'p': 0, 'e': 1})
            logging.info(f"Value Counts after encoding: {base_df[TARGET_COLUMN].value_counts()}")

            logging.info('Splitting base dataframe into Input and Target features')
            input_feature_df = base_df.drop(TARGET_COLUMN, axis=1)
            target_feature_df = base_df[TARGET_COLUMN]

            logging.info('Splitting Input and Target features into Train and Test sets')
            X_train, X_test, y_train, y_test = train_test_split(
                input_feature_df, target_feature_df, test_size=0.2, random_state=42)

            corr_threshold = config_entity.DataTransformationConfig.CORRELATION_THRESHOLD_VALUE

            logging.info(f"Getting input features which are at-least {corr_threshold}% associated with Target feature")
            associated_columns = get_associated_columns(base_df)

            preprocess_obj = self.get_data_transformation_object(associated_columns)

            logging.info("Pre-processing Train and Test DataFrame")
            X_train_arr = preprocess_obj.fit_transform(X_train)
            X_test_arr = preprocess_obj.transform(X_test)
            logging.info("Pre-processing done: got X_Train_arr & X_Test_arr Numpy arrays")

            logging.info('Concatenating x_train_arr and y_train to a training Numpy array dataset')
            train_arr = np.c_[X_train_arr, np.array(y_train)]

            logging.info('Concatenating X_test_arr and y_test to a testing Numpy array dataset')
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            # Saving train_arr and test_arr numpy arrays
            logging.info("Saving Train & Test Numpy arrays")
            save_numpy_array_data(path=self.data_transformation_config.train_np_arr_path, array=train_arr)
            save_numpy_array_data(path=self.data_transformation_config.test_np_arr_path, array=test_arr)

            # Saving pre-processor object path
            save_object(path=self.data_transformation_config.preprocessor_obj_path, obj=preprocess_obj)

            logging.info('Preparing Artifacts')
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                preprocessor_object_path=self.data_transformation_config.preprocessor_obj_path,
                train_arr_path=self.data_transformation_config.train_np_arr_path,
                test_arr_path=self.data_transformation_config.test_np_arr_path
            )

            logging.info("Data Transformation Completed.")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)


def get_associated_columns(dataframe: pd.DataFrame) -> Optional[list]:
    """
    This function checks for association/correlation of input features with target feature using Cramer's V rule.
    :param dataframe: base dataframe
    :return: A list of most associated columns
    """
    try:
        logging.info("Checking for Association (using CRAMER'S-V rule)")
        associations = dict()
        associated_columns = list()
        corr_threshold = config_entity.DataTransformationConfig.CORRELATION_THRESHOLD_VALUE
        for col_name in dataframe.columns:
            res = cramers_V(dataframe[TARGET_COLUMN], dataframe[col_name])
            associations[col_name] = round(res * 100, 2)

        for col_name, corr_value in associations.items():
            if corr_value >= corr_threshold:
                associated_columns.append(col_name)

        if TARGET_COLUMN in associated_columns:
            associated_columns.remove(TARGET_COLUMN)

        if len(associated_columns) > 0:  # there is always going to be TARGET_COLUMN in this list
            logging.info(f"Most associated columns with '{TARGET_COLUMN}' are : {associated_columns}")
            return associated_columns

        logging.info('NO ASSOCIATED COLUMNS FOUND.')
        return None

    except Exception as e:
        raise CustomException(e, sys)
