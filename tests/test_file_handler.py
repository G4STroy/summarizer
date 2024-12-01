import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from file_handler import FileHandler
from azure.core.exceptions import ResourceNotFoundError, AzureError
import pandas as pd
import io

class TestFileHandler(unittest.TestCase):
    @patch('file_handler.container_client')
    def test_upload_to_blob_storage(self, mock_container_client):
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        
        file = MagicMock()
        file.name = "test_file.xlsx"
        file.read.return_value = b"test content"
        
        result = FileHandler.upload_to_blob_storage(file)
        
        self.assertEqual(result, "test_file.xlsx")
        mock_blob_client.upload_blob.assert_called_once_with(b"test content", overwrite=True)

    @patch('file_handler.container_client')
    def test_read_excel_from_blob(self, mock_container_client):
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        
        test_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        excel_buffer = io.BytesIO()
        test_data.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        
        mock_blob_client.download_blob.return_value.readall.return_value = excel_buffer.getvalue()
        
        downloaded_data = FileHandler.read_excel_from_blob("test_file.xlsx")
        
        pd.testing.assert_frame_equal(downloaded_data, test_data)

        # Test file not found scenario
        mock_blob_client.download_blob.side_effect = ResourceNotFoundError("Blob not found")
        with self.assertRaises(FileNotFoundError):
            FileHandler.read_excel_from_blob('non_existent.xlsx')

    @patch('file_handler.container_client')
    def test_read_excel_from_blob_azure_error(self, mock_container_client):
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob.side_effect = AzureError("Azure error")

        with self.assertRaises(IOError):
            FileHandler.read_excel_from_blob("error_file.xlsx")

if __name__ == '__main__':
    unittest.main()
