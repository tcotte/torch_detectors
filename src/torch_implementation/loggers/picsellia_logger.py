import os
import uuid

from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import LogType


class PicselliaLogger:
    def __init__(self, path_env_file):
        self._path_env_file = path_env_file
        self._experiment = self.get_picsellia_experiment()

    def get_picsellia_experiment_link(self):
        client_id = self._client.id
        project_id = self.get_project_id_from_experiment()
        experiment_id = os.getenv('EXPERIMENT_ID')

        link = f'https://app.picsellia.com/{str(client_id)}/project/{str(project_id)}/experiment/{experiment_id}'
        return link

    def get_project_id_from_experiment(self) -> uuid.UUID:
        for project in self._client.list_projects():
            for experiment in project.list_experiments():
                if str(experiment.id) == os.getenv('EXPERIMENT_ID'):
                    return project.id

    def get_picsellia_experiment(self):
        load_dotenv(self._path_env_file)

        PICSELLIA_TOKEN = os.getenv('PICSELLIA_TOKEN')
        self._client = Client(PICSELLIA_TOKEN, organization_name=os.getenv('ORGANIZATION_NAME'))

        picsellia_experiment = self._client.get_experiment_by_id(os.getenv('EXPERIMENT_ID'))

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

        print(f"Successfully logged to Picsellia\n You can follow experiment here: "
              f"{self.get_picsellia_experiment_link()} ")

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
        self.store_model(model_path=os.path.join(path_saved_models, 'best.pth'), model_name='model-best')
        self.store_model(model_path=os.path.join(path_saved_models, 'latest.pth'), model_name='model-latest')

    def store_model(self, model_path: str, model_name: str) -> None:
        self._experiment.store(model_name, model_path, do_zip=True)

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


if __name__ == '__main__':
    pl = PicselliaLogger(path_env_file=r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\.env')
    print(pl.get_picsellia_experiment_link())
