import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import requests

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from llama3_llm import Llama3LLM

class TestLlama3LLM(unittest.TestCase):
    @patch('requests.post')
    def test_llm_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response

        llm = Llama3LLM()
        response = llm._call("Test prompt")

        self.assertEqual(response, "Test response")
        mock_post.assert_called_once()

        # Check that the API was called with the correct data
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs['json']['messages'][0]['content'], "Test prompt")
        self.assertEqual(call_kwargs['json']['max_tokens'], 2000)  # Check for increased max_tokens
        self.assertEqual(call_kwargs['headers']['Content-Type'], "application/json")
        self.assertEqual(call_kwargs['headers']['x-ms-version'], "2023-11-03")

    @patch('requests.post')
    def test_llm_call_failure(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("API call failed")
        
        llm = Llama3LLM()
        with self.assertRaises(ValueError):
            llm._call("Test prompt")

if __name__ == '__main__':
    unittest.main()
