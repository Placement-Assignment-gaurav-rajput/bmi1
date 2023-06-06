import sys
import pandas as pd
import json
from source_code.logger import logging
from source_code.utils import load_object
from source_code.exception import CustomException
from path_resolver import latest_model_path, latest_preprocessor_path


class PredictPipeline:
    def __int__(self): ...

    @staticmethod
    def decode_prediction(prediction) -> str:
        """
        :param prediction: Encoded Prediction value
        :return: Decoded Prediction
        """
        try:
            if prediction == 1:
                return " : Mushroom is Edible"
            elif prediction == 0:
                return " : Mushroom is Poisonous"
        except Exception as e:
            raise CustomException(e, sys)

    def model_predict(self, features: pd.DataFrame):
        """
        :param features: Input provided by user is collected as Pandas DataFrame
        :return: Predicted Value by Model is returned
        """
        try:
            trained_model_path = latest_model_path()
            logging.info(f"Latest Trained Model Path: {trained_model_path}")

            pre_processor_obj_path = latest_preprocessor_path()
            logging.info(f"Preprocessor object path: {pre_processor_obj_path}")

            logging.info('Loading Pre-Processor and Model objects')
            pre_processor = load_object(file_path=pre_processor_obj_path)
            model = load_object(file_path=trained_model_path)

            logging.info(f"Input Feature shape: {features.shape}")

            logging.info('Processing the input features provided by user')
            scaled_features = pre_processor.transform(features)

            predictions_val = model.predict(scaled_features)
            logging.info(f"Decoding Predictions made : {predictions_val}")

            predictions = self.decode_prediction(predictions_val)
            logging.info(f"Decoded Prediction: {predictions}")

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size,
                 gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                 stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type,
                 spore_print_color, population, habitat):
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        :return: Input data as Pandas DataFrame
        """
        try:
            custom_data_dict = {
                "cap-shape": [self.cap_shape],
                "cap-surface": [self.cap_surface],
                "cap-color": [self.cap_color],
                "bruises": [self.bruises],
                "odor": [self.odor],
                "gill-attachment": [self.gill_attachment],
                "gill-spacing": [self.gill_spacing],
                "gill-size": [self.gill_size],
                "gill-color": [self.gill_color],
                "stalk-shape": [self.stalk_shape],
                "stalk-root": [self.stalk_root],
                "stalk-surface-above-ring": [self.stalk_surface_above_ring],
                "stalk-surface-below-ring": [self.stalk_surface_below_ring],
                "stalk-color-above-ring": [self.stalk_color_above_ring],
                "stalk-color-below-ring": [self.stalk_color_below_ring],
                "veil-type": [self.veil_type],
                "veil-color": [self.veil_color],
                "ring-number": [self.ring_number],
                "ring-type": [self.ring_type],
                "spore-print-color": [self.spore_print_color],
                "population": [self.population],
                "habitat": [self.habitat]
            }
            with open('output.json', 'w', encoding='utf-8') as obj:
                json.dump(custom_data_dict, obj, ensure_ascii=False, indent=4)

            # return pd.DataFrame(custom_data_dict)
            return custom_data_dict

        except Exception as e:
            raise CustomException(e, sys)
