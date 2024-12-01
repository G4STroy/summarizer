from llama3_llm import Llama3LLM

class SentimentAnalyzer:
    def __init__(self, llm: Llama3LLM):
        """
        Initialize the SentimentAnalyzer with an LLM instance.
        
        Parameters:
        llm (Llama3LLM): An instance of the Llama3LLM class for language model interactions.
        """
        self.llm = llm

    def analyze(self, text: str) -> str:
        """
        Analyze the sentiment of the provided text.

        Parameters:
        text (str): The text to analyze.

        Returns:
        str: The sentiment analysis result.
        """
        prompt = f"Analyze the sentiment of the following text and categorize it as positive, negative, or neutral. Provide a brief explanation for your categorization:\n\n{text}"
        response = self.llm._call(prompt)
        return response
