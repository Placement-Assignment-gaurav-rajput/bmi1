import os
from datetime import datetime
from dataclasses import dataclass

ARTIFACTS = 'artifacts'
BASE_DATA_FILE = 'base_data.csv'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
PREPROCESSOR = 'preprocessor.pkl'
MODEL_NAME = 'model.pkl'
TARGET_COLUMN = 'class'

ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS, f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}")


@dataclass
class DataIngestionConfig:
    DATA_INGESTION_DIR = os.path.join(ARTIFACTS_DIR, 'Data_Ingestion')
    base_data_path: str = os.path.join(DATA_INGESTION_DIR, BASE_DATA_FILE)
    MISSING_VALUE_THRESHOLD = 30


@dataclass
class DataTransformationConfig:
    DATA_TRANSFORMATION_DIR = os.path.join(ARTIFACTS_DIR, 'Data_Transformation')
    preprocessor_obj_path = os.path.join(DATA_TRANSFORMATION_DIR, PREPROCESSOR)
    train_np_arr_path = os.path.join(DATA_TRANSFORMATION_DIR, TRAIN_FILE.replace('csv', 'npz'))
    test_np_arr_path = os.path.join(DATA_TRANSFORMATION_DIR, TEST_FILE.replace('csv', 'npz'))
    CORRELATION_THRESHOLD_VALUE = 20  # This value is a percentage
    TEST_DATA_SIZE = 0.2  # test data is set 20% of base data


@dataclass
class ModelTrainerConfig:
    MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'Trained_Model')
    trained_model_path: str = os.path.join(MODEL_DIR, MODEL_NAME)
    expected_f1_score = 0.7
    over_fitting_threshold = 0.1
