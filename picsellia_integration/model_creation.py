import os

from dotenv import load_dotenv
from picsellia import Client

if __name__ == "__main__":
    load_dotenv('../.env')

    api_token = os.getenv('PICSELLIA_TOKEN')
    organization_name = os.getenv('ORGANIZATION_NAME')

    client = Client(api_token, organization_name=organization_name)

    model_name = "RetinaNet"
    model_description = "PyTorch RetinaNet architecture"

    my_model = client.create_model(
      name=model_name,
      description=model_description,
    )
