import logging
from llama3_llm import Llama3LLM
from data_processor import DataProcessor

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, llm: Llama3LLM):
        """
        Initialize the Summarizer with an LLM instance.
        
        Parameters:
        llm (Llama3LLM): An instance of the Llama3LLM class for language model interactions.
        """
        self.llm = llm

    def summarize(self, data_processor: 'DataProcessor', group_or_entity: str, name: str) -> str:
        """
        Generate a summary for the specified group or entity using the data processor.

        Parameters:
        data_processor (DataProcessor): An instance of the DataProcessor class to generate the prompt.
        group_or_entity (str): The type of summary to generate (e.g., group or entity).
        name (str): The name of the group or entity.

        Returns:
        str: The generated summary from the language model.
        """
        try:
            logger.debug(f"Generating summary for {group_or_entity}: {name}")
            prompt = data_processor.generate_analysis_prompt(group_or_entity, name)
            logger.debug("Analysis prompt generated successfully")
            
            logger.debug("Calling LLM for response")
            response = self.llm._call(prompt)
            logger.debug("LLM response received")
            
            return response
        except Exception as e:
            logger.error(f"Error in summarize method: {str(e)}", exc_info=True)
            raise
