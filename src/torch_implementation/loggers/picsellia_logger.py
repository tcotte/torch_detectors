import os
import uuid
from datetime import datetime
from typing import Union

from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import LogType, JobStatus, ExperimentStatus


class PicselliaLogger:
    def __init__(self, path_env_file, create_new_experiment: bool = True, run_name: Union[str, None] = None):
        self._client: Union[None, Client] = None
        self._path_env_file = path_env_file

        if not create_new_experiment:
            self._experiment = self.get_picsellia_experiment()
        else:
            self.create_experiment(run_name=run_name)

    def get_picsellia_experiment_link(self):
        client_id = self._client.id
        project_id = self.get_project_id_from_experiment()
        experiment_id = self._experiment.id

        link = f'https://app.picsellia.com/{str(client_id)}/project/{str(project_id)}/experiment/{experiment_id}'
        return link

    def get_project_id_from_experiment(self) -> uuid.UUID:
        for project in self._client.list_projects():
            for experiment in project.list_experiments():
                if str(experiment.id) == os.getenv('EXPERIMENT_ID'):
                    return project.id

    def connect_to_picsellia_platform(self):
        PICSELLIA_TOKEN = os.getenv('PICSELLIA_TOKEN')
        self._client = Client(PICSELLIA_TOKEN, organization_name=os.getenv('ORGANIZATION_NAME'))

    def get_picsellia_experiment(self):
        load_dotenv(self._path_env_file)

        self.connect_to_picsellia_platform()

        if not os.getenv('TRAINING_DATASET_ID') is None:
            picsellia_experiment = self._client.get_experiment_by_id(os.getenv('EXPERIMENT_ID'))

            picsellia_experiment.delete_all_logs()
            picsellia_experiment.delete_all_artifacts()

            return picsellia_experiment

        else:
            self.create_experiment(run_name=None)
            return self._experiment

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

    def on_epoch_end(self, training_losses: dict, accuracies: dict, current_lr: float, validation_losses=None) -> None:
        if validation_losses is None:
            validation_losses = {}

        for key, value in validation_losses.items():
            self._experiment.log(name=f'Validation loss {key}', type=LogType.LINE, data=value)

        for key, value in training_losses.items():
            self._experiment.log(name=f'Training loss {key}', type=LogType.LINE, data=value)

        for key, value in accuracies.items():
            self._experiment.log(name=f'Validation {key}', type=LogType.LINE, data=value)

        self._experiment.log(name='Learning rate', type=LogType.LINE, data=current_lr)

    def on_train_end(self, best_validation_map: float, path_saved_models: str):
        self._experiment.log(name="Best Validation Map", type=LogType.VALUE, data=best_validation_map)
        self.store_model(model_path=os.path.join(path_saved_models, 'best.pth'), model_name='model-best')
        self.store_model(model_path=os.path.join(path_saved_models, 'latest.pth'), model_name='model-latest')

        self._experiment.update(status=ExperimentStatus.TERMINATED)

    def store_model(self, model_path: str, model_name: str) -> None:
        self._experiment.store(model_name, model_path, do_zip=True)

    def create_experiment(self, run_name: Union[str, None] = None) -> None:
        load_dotenv(self._path_env_file)

        self.connect_to_picsellia_platform()

        project = self._client.get_project_by_id(os.getenv('PROJECT_ID'))

        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d-%H%M')

        self._experiment = project.create_experiment(name=run_name)

        if not os.getenv('TRAINING_DATASET_ID') is None:
            training_dataset_version = self._client.get_dataset_version_by_id(os.getenv('TRAINING_DATASET_ID'))
            self._experiment.attach_dataset(name="training", dataset_version=training_dataset_version)
            print(f'Attach {training_dataset_version.name} to {self._experiment.name}')

# if __name__ == '__main__':
#     pl = PicselliaLogger(path_env_file=r'C:\Users\tristan_cotte\PycharmProjects\yolov8_keras\.env')
#     print(pl.get_picsellia_experiment_link())
