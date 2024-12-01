import sys
import os
import unittest

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Group Names': ['Group1', 'Group2'],
            'Entity Name': ['Entity1', 'Entity2'],
            'Capability Name': ['Capability1', 'Capability2'],
            'Template Name': ['Template1', 'Template2'],
            'Assessment Date': ['2023-01-01', '2023-01-02'],
            'Assessment Number': [1, 2],
            'Rating': [4.0, 3.5],
            'Notes': ['Note1', 'Note2'],
            'Criteria': ['Criteria1', 'Criteria2'],
            'Criteria Stage': ['Stage1', 'Stage2']
        })
        self.processor = DataProcessor(self.data)

    def test_validate_data(self):
        # Test for missing columns
        data = self.data.drop(columns=['Group Names'])
        with self.assertRaises(ValueError):
            DataProcessor(data)

    def test_get_entities(self):
        entities = self.processor.get_entities()
        self.assertEqual(entities, ['Entity1', 'Entity2'])

    def test_get_groups(self):
        groups = self.processor.get_groups()
        self.assertEqual(groups, ['Group1', 'Group2'])

    def test_get_assessment_data(self):
        data = self.processor.get_assessment_data('Entity1')
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['Entity Name'], 'Entity1')

    def test_get_progress(self):
        progress = self.processor.get_progress('Entity1')
        self.assertEqual(progress.iloc[0], 4.0)

    def test_get_capability_scores(self):
        scores = self.processor.get_capability_scores('Entity1')
        self.assertEqual(scores['Capability1'], 4.0)

    def test_get_criteria_distribution(self):
        distribution = self.processor.get_criteria_distribution('Entity1')
        self.assertEqual(distribution['Stage1'], 1)

    def test_get_notes(self):
        notes = self.processor.get_notes('Entity1')
        self.assertEqual(notes, ['Note1'])

    def test_get_assessment_dates(self):
        dates = self.processor.get_assessment_dates('Entity1')
        self.assertEqual(dates, ['2023-01-01'])

    def test_get_template_names(self):
        templates = self.processor.get_template_names('Entity1')
        self.assertEqual(templates, ['Template1'])

if __name__ == '__main__':
    unittest.main()