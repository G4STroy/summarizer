import os
import io
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from azure.core.exceptions import ResourceNotFoundError, AzureError

# Load environment variables
load_dotenv()

AZURE_BLOB_ACCOUNT_URL = os.getenv("AZURE_BLOB_ACCOUNT_URL")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

# Create clients at the module level
blob_service_client = BlobServiceClient(account_url=AZURE_BLOB_ACCOUNT_URL, credential=DefaultAzureCredential())
container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

class FileHandler:
    @staticmethod
    def upload_to_blob_storage(file):
        """
        Upload a file to Azure Blob Storage.

        Parameters:
        file: The file object to upload.

        Returns:
        str: The name of the uploaded file.
        """
        blob_client = container_client.get_blob_client(file.name)
        file_contents = file.read()
        blob_client.upload_blob(file_contents, overwrite=True)
        return file.name

    @staticmethod
    def read_excel_from_blob(file_name):
        """
        Read an Excel file from Azure Blob Storage.

        Parameters:
        file_name (str): The name of the file to read.

        Returns:
        pd.DataFrame: The contents of the Excel file as a pandas DataFrame.

        Raises:
        FileNotFoundError: If the file is not found in the blob storage.
        IOError: If an error occurs while reading the file from blob storage.
        """
        try:
            blob_client = container_client.get_blob_client(file_name)
            blob_data = blob_client.download_blob()
            excel_data = io.BytesIO(blob_data.readall())
            return pd.read_excel(excel_data)
        except ResourceNotFoundError:
            raise FileNotFoundError(f"The file {file_name} was not found in the blob storage.")
        except AzureError as e:
            raise IOError(f"An error occurred while reading the file from blob storage: {str(e)}")
