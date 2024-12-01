import pandas as pd
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataProcessor with a DataFrame and validate it.
        
        Parameters:
        data (pd.DataFrame): The DataFrame containing assessment data.
        """
        self.data = data
        self._validate_data()

    def _validate_data(self):
        """
        Validate that the DataFrame contains all required columns.
        
        Raises:
        ValueError: If any required columns are missing from the DataFrame.
        """
        required_columns = [
            'Group Names', 'Entity Name', 'Capability Name', 'Template Name',
            'Assessment Date', 'Assessment Number', 'Rating', 'Notes',
            'Criteria', 'Criteria Stage'
        ]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def get_entities(self) -> List[str]:
        """
        Retrieve a list of unique entity names from the data.

        Returns:
        List[str]: A list of unique entity names.
        """
        logger.debug("Retrieving entities")
        return self.data['Entity Name'].unique().tolist()

    def get_groups(self) -> List[str]:
        """
        Retrieve a list of unique group names from the data.

        Returns:
        List[str]: A list of unique group names.
        """
        logger.debug("Retrieving groups")
        return self.data['Group Names'].unique().tolist()

    def get_assessment_data(self, entity_or_group: str, is_group: bool = False) -> List[Dict]:
        """
        Retrieve assessment data for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        List[Dict]: A list of dictionaries containing the assessment data.
        """
        logger.debug(f"Getting assessment data for {'group' if is_group else 'entity'}: {entity_or_group}")
        if is_group:
            data = self.data[self.data['Group Names'] == entity_or_group]
        else:
            data = self.data[self.data['Entity Name'] == entity_or_group]
        
        result = data.to_dict('records')
        logger.debug(f"Assessment data retrieved. Number of records: {len(result)}")
        return result

    def get_progress(self, entity_or_group: str, is_group: bool = False) -> pd.Series:
        """
        Retrieve progress data for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        pd.Series: A series containing the progress data.
        """
        logger.debug(f"Getting progress for {'group' if is_group else 'entity'}: {entity_or_group}")
        data = pd.DataFrame(self.get_assessment_data(entity_or_group, is_group))
        return data.groupby('Assessment Number')['Rating'].mean()

    def get_capability_scores(self, entity_or_group: str, is_group: bool = False) -> Dict[str, float]:
        """
        Retrieve capability scores for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        Dict[str, float]: A dictionary containing the capability scores.
        """
        logger.debug(f"Getting capability scores for {'group' if is_group else 'entity'}: {entity_or_group}")
        data = pd.DataFrame(self.get_assessment_data(entity_or_group, is_group))
        return data.groupby('Capability Name')['Rating'].mean().to_dict()

    def get_criteria_distribution(self, entity_or_group: str, is_group: bool = False) -> Dict[str, int]:
        """
        Retrieve criteria distribution for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        Dict[str, int]: A dictionary containing the criteria distribution.
        """
        logger.debug(f"Getting criteria distribution for {'group' if is_group else 'entity'}: {entity_or_group}")
        data = pd.DataFrame(self.get_assessment_data(entity_or_group, is_group))
        return data['Criteria Stage'].value_counts().to_dict()

    def get_notes(self, entity_or_group: str, is_group: bool = False) -> List[str]:
        """
        Retrieve notes for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        List[str]: A list of notes.
        """
        logger.debug(f"Getting notes for {'group' if is_group else 'entity'}: {entity_or_group}")
        data = pd.DataFrame(self.get_assessment_data(entity_or_group, is_group))
        return data['Notes'].dropna().tolist()

    def get_assessment_dates(self, entity_or_group: str, is_group: bool = False) -> List[str]:
        """
        Retrieve assessment dates for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        List[str]: A list of unique assessment dates.
        """
        logger.debug(f"Getting assessment dates for {'group' if is_group else 'entity'}: {entity_or_group}")
        data = pd.DataFrame(self.get_assessment_data(entity_or_group, is_group))
        return data['Assessment Date'].unique().tolist()

    def get_template_names(self, entity_or_group: str, is_group: bool = False) -> List[str]:
        """
        Retrieve template names for a specific entity or group.

        Parameters:
        entity_or_group (str): The name of the entity or group.
        is_group (bool): Whether to filter by group or entity.

        Returns:
        List[str]: A list of unique template names.
        """
        logger.debug(f"Getting template names for {'group' if is_group else 'entity'}: {entity_or_group}")
        data = pd.DataFrame(self.get_assessment_data(entity_or_group, is_group))
        return data['Template Name'].unique().tolist()
    
    def generate_analysis_prompt(self, group_or_entity: str, name: str) -> str:
        """
        Generate a detailed analysis prompt based on the data for a specific group or entity.

        Parameters:
        group_or_entity (str): The type of analysis to generate (e.g., group or entity).
        name (str): The name of the group or entity.

        Returns:
        str: The generated analysis prompt.
        """
        logger.debug(f"Generating analysis prompt for {group_or_entity}: {name}")
        assessment_data = pd.DataFrame(self.get_assessment_data(name, is_group=(group_or_entity == "Group")))
        
        prompt = f"""
        Analyze the following assessment data for {group_or_entity}: {name}
        Group: {assessment_data['Group Names'].iloc[0]}

        Template Name(s): {', '.join(self.get_template_names(name, group_or_entity == "Group"))}
        Assessment Date(s): {', '.join(self.get_assessment_dates(name, group_or_entity == "Group"))}
        Assessment Number(s): {', '.join(map(str, assessment_data['Assessment Number'].unique()))}
        Total Number of Assessments Analyzed: {len(assessment_data['Assessment Number'].unique())}

        Please provide the following analysis in this order:

        1. Comprehensive Notes Summary and Sentiment Analysis:
        Start with a detailed summary of all notes across all capabilities and assessments. This should include:
        - A chronological overview of the notes, highlighting key themes and changes over time
        - Direct quotes from the notes that illustrate important points or shifts in focus
        - An analysis of the overall sentiment in the notes, including how it has changed over time
        - Specific examples of positive and negative sentiments, supported by quotes
        - An interpretation of what these sentiments suggest about the entity's progress and challenges
        If there are no notes for a particular capability or assessment, mention this fact in your analysis.

        2. A concise summary of the overall performance across all capabilities, highlighting key improvements and areas of concern.

        3. A focused analysis of progress over time, mentioning only capabilities with significant changes.

        4. Top 3 strengths and top 3 areas for improvement, based on the most recent ratings and progress over time.

        5. Detailed Analysis of Capabilities with Notes:
        For each capability with notes, provide:
        - The capability name and its most recent rating
        - A chronological summary of all notes for this capability, including direct quotes
        - Your interpretation of how the notes relate to the rating and how they've changed over time
        - A specific recommendation based on the notes and rating

        6. Summary Analysis of Capabilities without Notes:
        For capabilities without notes, provide:
        - The capability name and its most recent rating
        - An analysis based on the criteria for the current score and what's needed for a higher score
        - A specific recommendation for improvement

        7. 3-5 specific, actionable recommendations for future focus areas, based on the identified weaknesses and the content of the notes.

        Capabilities with notes:
        """

        for capability, group in assessment_data.groupby('Capability Name'):
            notes = group[['Assessment Number', 'Assessment Date', 'Notes']].dropna(subset=['Notes'])
            if not notes.empty:
                prompt += f"""
                Capability: {capability}
                Most Recent Rating: {group['Rating'].iloc[-1]:.2f}
                Notes Over Time:
                """
                for _, row in notes.iterrows():
                    prompt += f"""
                    Assessment Number: {row['Assessment Number']}
                    Date: {row['Assessment Date']}
                    Notes: {row['Notes']}
                    """
                prompt += f"""
                Criteria: {'; '.join(group['Criteria'].unique())}
                Criteria Stage: {', '.join(group['Criteria Stage'].unique())}

                """

        prompt += """
        Capabilities without notes:
        """

        for capability, group in assessment_data.groupby('Capability Name'):
            if group['Notes'].isna().all():
                prompt += f"""
                Capability: {capability}
                Most Recent Rating: {group['Rating'].iloc[-1]:.2f}
                Criteria: {'; '.join(group['Criteria'].unique())}
                Criteria Stage: {', '.join(group['Criteria Stage'].unique())}

                """

        prompt += """
        Please provide a focused analysis based on this data, avoiding redundancies and emphasizing insights from the notes and criteria. 
        Ensure that your analysis includes specific ratings, meaningful quotes from notes where available, and clear, actionable recommendations.
        For capabilities without notes, base your analysis on the criteria and current rating, explaining what's needed for improvement.
        
        In the Comprehensive Notes Summary and Sentiment Analysis section, provide a detailed overview of all notes, their sentiment, and how they've evolved over time. 
        Use specific quotes to support your analysis and highlight any significant trends or changes in sentiment across different capabilities and assessments.
        If a capability has no notes, include this information in your analysis and consider what this lack of notes might imply.
        """

        logger.debug("Analysis prompt generated successfully")
        return prompt
