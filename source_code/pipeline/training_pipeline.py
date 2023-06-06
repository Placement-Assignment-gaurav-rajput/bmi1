import os
import sys
from source_code.components.data_ingestion import DataIngestion
from source_code.components.data_transformation import DataTransformation
from source_code.components.model_trainer import ModelTrainer
from source_code.exception import CustomException


def start_training_pipeline():
    try:
        os.chdir("../..")
        # Data-Ingestion
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data-Transformation
        data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifacts = data_transform.initiate_data_transformation()

        # Model-Training
        model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts)
        model_trainer_artifact = model_trainer.initiate_model_training()

        model_obj = model_trainer_artifact.trained_model_path
        preprocessor_obj = data_transformation_artifacts.preprocessor_object_path

        return model_obj, preprocessor_obj

    except Exception as e:
        raise CustomException(e, sys)
