import os

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import AzureBlobStorageContainerLoader



load_dotenv()


directory_path = "Data"




blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

def write_files_to_blob():
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            blob_name = os.path.relpath(file_path, directory_path)

            blob_client = blob_service_client.get_blob_client(os.environ.get('CONTAINER_NAME'), blob=blob_name)

            with open(file_path, "rb") as data:
                blob_client.upload_blob(data)

                print(f"{file_path} -> {blob_name} uploaded")


def get_files(prefix):
    loader = AzureBlobStorageContainerLoader(
            conn_str=os.environ.get('STORAGE_CONNECTION_STRING'),
            container=os.environ.get('CONTAINER_NAME'),
            #prefix=prefix
    )
    docs = loader.load()
    return docs

