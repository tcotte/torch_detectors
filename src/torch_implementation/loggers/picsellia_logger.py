import os

from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import LogType


class PicselliaLogger:
    def __init__(self, path_env_file):
        self._path_env_file = path_env_file
        self._experiment = self.get_picsellia_experiment()

    def get_picsellia_experiment(self):
        load_dotenv(self._path_env_file)

        PICSELLIA_TOKEN = os.getenv('PICSELLIA_TOKEN')
        client = Client(PICSELLIA_TOKEN, organization_name=os.getenv('ORGANIZATION_NAME'))

        picsellia_experiment = client.get_experiment_by_id(os.getenv('EXPERIMENT_ID'))

        picsellia_experiment.delete_all_logs()
        picsellia_experiment.delete_all_artifacts()

        return picsellia_experiment

    def log_split_table(self, annotations_in_split: dict, title: str):
        data = {'x': [], 'y': []}
        for key, value in annotations_in_split.items():
            data['x'].append(key)
            data['y'].append(value)

        self._experiment.log(name=title, type=LogType.BAR, data=data)

    def on_train_begin(self, params, class_mapping):
        self._experiment.log(name='Parameters', type=LogType.TABLE, data=params)
        self._experiment.log(name='LabelMap', type=LogType.TABLE,
                                 data={str(key): value for key, value in class_mapping.items()})

    #     if self._config_file is not None:
    #         self._picsellia_experiment.store('config', self._config_file)

    def on_epoch_end(self, losses: dict, accuracies: dict, current_lr: float) -> None:
        for key, value in losses.items():
            self._experiment.log(name=f'Training {key}', type=LogType.LINE, data=value)

        for key, value in accuracies.items():
            self._experiment.log(name=f'Validation {key}', type=LogType.LINE, data=value)

        self._experiment.log(name='Learning rate', type=LogType.LINE, data=current_lr)

    def on_train_end(self, best_validation_map: float, path_saved_models: str):
        self._experiment.log(name="Best Validation Map", type=LogType.VALUE, data=best_validation_map)
        self.store_model(model_path=os.path.join(path_saved_models, 'best.pth'), model_name='best')
        self.store_model(model_path=os.path.join(path_saved_models, 'latest.pth'), model_name='latest')

    def store_model(self, model_path: str, model_name: str) -> None:
        self._experiment.create_artifact(
            name=model_name, filename=model_path, object_name=os.path.basename(model_path), large=True
        )

    # def on_train_end(self, logs=None):
    #     self._picsellia_experiment.log(name="Best Validation Map", type=LogType.VALUE, value=self.best_map)
    #     self.store_latest_model()
    #
    # def store_latest_model(self):
    #     latest_model_object_name = 'latest-model.h5'
    #     latest_model_path = os.path.join(os.path.dirname(self.save_path), latest_model_object_name)
    #     self.model.save(latest_model_path)
    #     self._picsellia_experiment.create_artifact(
    #         name="model_latest", filename=latest_model_path, object_name=latest_model_object_name, large=True
    #     )