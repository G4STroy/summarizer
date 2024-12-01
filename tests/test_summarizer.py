import sys
import os
import unittest
from unittest.mock import MagicMock

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from summarizer import Summarizer
from llama3_llm import Llama3LLM
from data_processor import DataProcessor

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=Llama3LLM)
        self.summarizer = Summarizer(self.mock_llm)
        self.mock_data_processor = MagicMock(spec=DataProcessor)

    def test_summarize(self):
        self.mock_llm._call.return_value = "Test comprehensive analysis"
        self.mock_data_processor.generate_analysis_prompt.return_value = "Test prompt"
        
        result = self.summarizer.summarize(self.mock_data_processor, "Group", "TestGroup")
        
        self.assertEqual(result, "Test comprehensive analysis")
        self.mock_data_processor.generate_analysis_prompt.assert_called_once_with("Group", "TestGroup")
        self.mock_llm._call.assert_called_once_with("Test prompt")

if __name__ == '__main__':
    unittest.main()
