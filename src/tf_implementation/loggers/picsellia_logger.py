from dotenv import load_dotenv
import os

from picsellia import Client, Experiment
from picsellia.types.enums import LogType


def get_picsellia_experiment():
    load_dotenv()

    PICSELLIA_TOKEN = os.getenv('PICSELLIA_TOKEN')
    client = Client(PICSELLIA_TOKEN, organization_name=os.getenv('ORGANIZATION_NAME'))

    picsellia_experiment = client.get_experiment_by_id(os.getenv('EXPERIMENT_ID'))

    picsellia_experiment.delete_all_logs()
    picsellia_experiment.delete_all_artifacts()

    return picsellia_experiment

def log_split_table(picsellia_experiment: Experiment, annotations_in_split: dict, title: str):
    data = {'x': [], 'y': []}
    for key, value in annotations_in_split.items():
        data['x'].append(key)
        data['y'].append(value)

    picsellia_experiment.log(name=title, type=LogType.BAR, data=data)

if __name__ == '__main__':
    experiment = get_experiment()
    data = [8]
    experiment.log('TotalLoss', data, LogType.LINE)
    # experiment.log('TotalLoss', data, LogType.LINE)
