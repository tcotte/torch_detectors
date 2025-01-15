import os
from typing import Union

import keras
import keras_cv
from dotenv import load_dotenv
from picsellia import Client, Experiment
from picsellia.types.enums import LogType


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path, picsellia_experiment: Union[Experiment, None], config_file: Union[str, None] = None):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

        if picsellia_experiment is None:
            self._picsellia_experiment = self.get_picsellia_experiment()
        else:
            self._picsellia_experiment = picsellia_experiment

        self._config_file = config_file

    def on_train_begin(self, logs=None):
        if self._config_file is not None:
            self._picsellia_experiment.store('config', self._config_file)
            # TODO write parameters in log

    def get_picsellia_experiment(self):
        load_dotenv()

        PICSELLIA_TOKEN = os.getenv('PICSELLIA_TOKEN')
        client = Client(PICSELLIA_TOKEN, organization_name=os.getenv('ORGANIZATION_NAME'))

        picsellia_experiment = client.get_experiment_by_id(os.getenv('EXPERIMENT_ID'))

        picsellia_experiment.delete_all_logs()
        picsellia_experiment.delete_all_artifacts()

        return picsellia_experiment

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result()
        logs.update(metrics)

        self._picsellia_experiment.log('Total Valid Loss', logs['loss'], LogType.LINE)
        self._picsellia_experiment.log('Class Valid Loss', logs['class_loss'], LogType.LINE)
        self._picsellia_experiment.log('Box Valid Loss', logs['box_loss'], LogType.LINE)

        self._picsellia_experiment.log('Valid MaP', float(logs['MaP'].numpy()), LogType.LINE)
        self._picsellia_experiment.log('Valid MaP IoU=0.5', float(logs['MaP@[IoU=50]'].numpy()), LogType.LINE)
        self._picsellia_experiment.log('Valid MaP IoU=0.75', float(logs['MaP@[IoU=75]'].numpy()), LogType.LINE)
        self._picsellia_experiment.log('Valid Recall [MaxDet=100]', float(logs['Recall@[max_detections=100]'].numpy()),
                                       LogType.LINE)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs

    def on_train_end(self, logs=None):
        self._picsellia_experiment.log(name="Best Validation Map", type=LogType.VALUE, value=self.best_map)
        self.store_latest_model()

    def store_latest_model(self):
        latest_model_object_name = 'latest-model.h5'
        latest_model_path = os.path.join(os.path.dirname(self.save_path), latest_model_object_name)
        self.model.save(latest_model_path)
        self._picsellia_experiment.create_artifact(
            name="model_latest", filename=latest_model_path, object_name=latest_model_object_name, large=True
        )
