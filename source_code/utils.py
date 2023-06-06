import os
import sys
import pickle
import pymongo
import numpy as np
import pandas as pd
from typing import List
import scipy.stats as ss
from scipy.stats import chi2_contingency

from source_code.logger import logging
from source_code.exception import CustomException
from source_code.entity.config_entity import DataIngestionConfig

MONGO_DB_URL = os.environ.get('MONGO_DB_URL')

mongo_client = pymongo.MongoClient(MONGO_DB_URL)


def get_df_from_mongo(db_name: str, collection_name: str) -> pd.DataFrame:
    """
    This function reads data from Mongo Atlas and return it as Pandas Dataframe.
    :param db_name: Mongo database name.
    :param collection_name: Mongo database collection name.
    :return: Pandas DataFrame
    """
    try:
        dataframe = pd.DataFrame(list(mongo_client[db_name][collection_name].find()))
        logging.info(f"Dataset Shape = {dataframe.shape}")
        if "_id" in dataframe.columns:
            dataframe = dataframe.drop("_id", axis=1)
        return dataframe
    except Exception as e:
        raise CustomException(e, sys)


def drop_missing_val_columns(df: pd.DataFrame) -> List[str]:
    """
    This function takes pandas DataFrame as input parameter and returns a list of columns which are below a
    specified threshold.
    :param df: pandas DataFrame.
    :return: list of columns to be deleted.
    """
    try:
        obj = DataIngestionConfig()
        # df = df.replace(to_replace='?', value=np.nan)
        missing_data_dict = dict()
        drop_col_list = list()
        for col_name in df.columns:
            missing_data_dict[col_name] = (df[col_name].isna().sum() / len(df)) * 100

        for col_name, missing_value in missing_data_dict.items():
            if missing_value > obj.MISSING_VALUE_THRESHOLD:
                drop_col_list.append(col_name)

        return drop_col_list
    except Exception as e:
        raise CustomException(e, sys)


def drop_single_unique_val_columns(df: pd.DataFrame) -> List[str]:
    """
    :param df: Pandas DataFrame
    :return: List of columns with only one unique value.
    """
    try:
        column_name_with_unique_val_count = dict()
        unique_val_column_list = list()

        for col_name in df.columns:
            count = 0
            for _ in df[col_name].unique():
                count += 1
            column_name_with_unique_val_count[col_name] = count

        for col_name, unique_val in column_name_with_unique_val_count.items():
            if unique_val == 1:
                unique_val_column_list.append(col_name)

        return unique_val_column_list
    except Exception as e:
        raise CustomException(e, sys)


def cramers_V(target_column: pd.DataFrame, col_name: pd.DataFrame):
    """
    This function checks for association between target categorical columns and other categorical columns.
    :param target_column: Target Column
    :param col_name: Column name with which association/correlation needs to be calculated.
    :return: association value. ( <5 is optimum )
    """
    try:
        crosstab = np.array(pd.crosstab(target_column, col_name, rownames=None, colnames=None))  # Cross table building
        stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
        total_obs = np.sum(crosstab)  # Number of observations
        mininum_val = min(crosstab.shape) - 1  # Take the minimum value between columns and the rows of the cross table
        return stat / (total_obs * mininum_val)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(path, obj):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(path: str, array: np.array):
    """
    :param path: takes file path where the numpy array will be saved.
    :param array: Numpy array to save.
    :return: NA
    """
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as file_obj:
            np.save(file=file_obj, arr=array)
            # np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_numpy_array_data(path: str) -> np.array:
    """
    This function loads numpy array from a specified location.
    :param path: Path of the numpy array to loaded from.
    :return: Numpy array.
    """
    try:
        with open(path, "rb") as file_obj:
            return np.load(file=file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
