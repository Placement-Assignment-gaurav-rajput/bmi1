import sys

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from source_code.logger import logging
from source_code.exception import CustomException
from source_code.entity.config_entity import ModelTrainerConfig
from source_code.utils import load_numpy_array_data, save_object
from source_code.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifact):
        logging.info(f"{'--'*20} Model Training {20*'--'}")
        self.data_transformation_artifact = data_transformation_artifacts
        self.model_trainer_config = ModelTrainerConfig()

    @staticmethod
    def train_model(X, y):
        """
        This function trains machine learning model using LogisticRegression()
        :param X: Training input dataset
        :param y: Target input dataset
        :return: Training model object of LogisticRegression()
        """
        try:
            clf = LogisticRegression()
            logging.info(f"Training the Model with LogisticRegression()")
            clf.fit(X, y)
            return clf
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        logging.info('Model Training initiated')
        try:
            logging.info('Loading train and test numpy arrays')
            train_arr = load_numpy_array_data(self.data_transformation_artifact.train_arr_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.test_arr_path)

            logging.info('Splitting input and target feature from both train and test array')
            X_train, y_train = train_arr[:, :-1],  train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Training model using LogisticRegression()
            model = self.train_model(X=X_train, y=y_train)

            logging.info('Making Predictions for y_train and y_test values')
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            logging.info('Calculating f1_score for predicted values of y_train & y_test')
            train_f1_score = f1_score(y_true=y_train, y_pred=y_train_pred)
            test_f1_score = f1_score(y_true=y_test, y_pred=y_test_pred)

            logging.info(f"Train f1_score: {train_f1_score}, Test f1_score: {test_f1_score}")

            expected_f1_score = self.model_trainer_config.expected_f1_score

            # Checking for UNDER-FITTING
            logging.info("Checking for Under-fitting...")
            if test_f1_score < expected_f1_score:
                logging.info("MODEL UNDER-FITTING :(")
                raise Exception(f"MODEL SUB-PAR, Expected Accuracy({expected_f1_score}) > Model Accuracy({test_f1_score})")
            else:
                logging.info('No Under-fitting detected')

            # Checking for OVER-FITTING
            logging.info("Checking for Over-fitting...")
            overfit_threshold = self.model_trainer_config.over_fitting_threshold
            diff = abs(train_f1_score - test_f1_score)
            if diff > overfit_threshold:
                logging.info(f"MODEL OVER-FITTING :( Over-fit Threshold of {overfit_threshold * 100}% crossed")
                raise Exception(f"Over-fitting threshold {overfit_threshold * 100}% is crossed. Model Over-fitting")
            else:
                logging.info('No Over-fitting detected')

            logging.info('Saving Model Object')
            save_object(path=self.model_trainer_config.trained_model_path, obj=model)

            logging.info('Preparing Artifacts')
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info('Model Training Completed.')
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
