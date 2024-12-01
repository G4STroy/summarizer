import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from file_handler import FileHandler
from data_processor import DataProcessor
from sentiment_analyzer import SentimentAnalyzer
from summarizer import Summarizer
from llama3_llm import Llama3LLM
from azure.core.exceptions import ResourceNotFoundError, AzureError
import pandas as pd
import io

class TestIntegration(unittest.TestCase):
    @patch('file_handler.container_client')
    def test_file_handler_integration(self, mock_container_client):
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

    @patch('llama3_llm.requests.post')
    def test_llama3_api_integration(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_post.return_value = mock_response

        llm = Llama3LLM()
        test_prompt = "Summarize this text: The quick brown fox jumps over the lazy dog."
        response = llm._call(test_prompt)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        mock_post.assert_called_once()

    @patch('file_handler.container_client')
    def test_read_excel_from_blob_not_found(self, mock_container_client):
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob.side_effect = ResourceNotFoundError("Blob not found")

        with self.assertRaises(FileNotFoundError):
            FileHandler.read_excel_from_blob("non_existent_file.xlsx")

    @patch('file_handler.container_client')
    def test_read_excel_from_blob_azure_error(self, mock_container_client):
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob.side_effect = AzureError("Azure error")

        with self.assertRaises(IOError):
            FileHandler.read_excel_from_blob("error_file.xlsx")

if __name__ == '__main__':
    unittest.main()
