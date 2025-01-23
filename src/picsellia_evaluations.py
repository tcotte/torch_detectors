import os

from dotenv import load_dotenv
from picsellia import Client

load_dotenv('../.env')




if __name__ == '__main__':
    EXPERIMENT_ID = '019493b6-8e82-73ab-bb93-067dd78ecb18'

    picsellia_token = os.getenv('PICSELLIA_TOKEN')
    organisation_name = os.getenv('ORGANISATION_NAME')
    picsellia_client = Client(picsellia_token, organization_name=os.getenv('ORGANIZATION_NAME'))

    experiment = picsellia_client.get_experiment_by_id(EXPERIMENT_ID)

    print(experiment)