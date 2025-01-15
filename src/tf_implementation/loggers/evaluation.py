import os
import random

from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import InferenceType

if __name__ == "__main__":
    load_dotenv()

    PICSELLIA_TOKEN = os.getenv('PICSELLIA_TOKEN')

    client = Client(PICSELLIA_TOKEN, organization_name=os.getenv('ORGANIZATION_NAME'))

    experiment = client.get_experiment_by_id(os.getenv('EXPERIMENT_ID'))

    dataset_version = client.get_dataset_version_by_id(os.getenv("DATASET_VERSION_ID")) #the id of the dataset version is availble on the UI

    labels = dataset_version.list_labels()

    if experiment.list_attached_dataset_versions() == []:
        experiment.attach_dataset(name='initial', dataset_version=dataset_version)

    ##this recipe is showing how to add and compute an evaluation for one asset, but for sure you can loop on all the assets of your test set
    for idx in range(2):
        asset = dataset_version.find_asset(filename=dataset_version.list_assets()[idx].filename) #the asset can be retrieved through several methods

        gt_annotation = asset.get_annotation()
        rectangles = gt_annotation.list_rectangles()

        predicted_rectangles = []

        for rectangle in rectangles:
            offset_rand_x = random.randint(0, 10)
            offset_rand_y = random.randint(0, 10)
            rand_confidence = random.randint(0, 100)

            rec = (rectangle.x + offset_rand_x, rectangle.y + offset_rand_y, rectangle.w, rectangle.h, rectangle.label,
                   rand_confidence/100)

            predicted_rectangles.append(rec)




        ##in this case the evaluation is done in Object Detection mode, and the predicitons are prompted manually


        evaluation = experiment.add_evaluation(asset, rectangles=predicted_rectangles)

        job = experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)
    job.wait_for_done()

