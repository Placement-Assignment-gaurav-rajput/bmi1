from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    base_data_path: str
    # train_file_path: str
    # test_file_path: str


@dataclass
class DataTransformationArtifact:
    preprocessor_object_path: str
    train_arr_path: str
    test_arr_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
