import os
import requests
import logging
from pydantic import BaseModel
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

LLAMA3_API_ENDPOINT = os.getenv("LLAMA3_API_ENDPOINT")
LLAMA3_API_KEY = os.getenv("LLAMA3_API_KEY")

class Llama3LLM(LLM, BaseModel):
    """
    A class to interact with the Llama3 language model API.
    """
    endpoint: str = LLAMA3_API_ENDPOINT
    api_key: str = LLAMA3_API_KEY

    def __init__(self, **data: Any):
        """
        Initialize the Llama3LLM with endpoint and API key.

        Raises:
        ValueError: If endpoint or API key is not set.
        """
        super().__init__(**data)
        if not self.endpoint or not self.api_key:
            raise ValueError("LLAMA3_API_ENDPOINT and LLAMA3_API_KEY environment variables must be set")

    @property
    def _llm_type(self) -> str:
        """
        Return the type of the language model.

        Returns:
        str: The type of the language model.
        """
        return "Llama3"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        """
        Make a call to the Llama3 API with the provided prompt.

        Parameters:
        prompt (str): The prompt to send to the Llama3 API.
        stop (Optional[List[str]]): List of stop sequences for the API call.
        run_manager (Optional[CallbackManagerForLLMRun]): Optional callback manager for the API call.

        Returns:
        str: The response from the Llama3 API.

        Raises:
        ValueError: If the API call fails.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-ms-version": "2023-11-03"  # Use the latest API version
        }
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        response = None
        try:
            response = requests.post(self.endpoint, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {str(e)}")
            if response:
                if response.status_code == 400:
                    raise ValueError("Bad request. Please check the parameters.")
                elif response.status_code == 401:
                    raise ValueError("Authentication failed. Please check your API key.")
                elif response.status_code == 429:
                    raise ValueError("Rate limit exceeded. Please try again later.")
                elif response.status_code == 500:
                    raise ValueError("Internal server error. Please try again later.")
            raise ValueError(f"API call failed with an unknown error: {str(e)}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Return the identifying parameters of the Llama3LLM instance.

        Returns:
        Mapping[str, Any]: The identifying parameters of the Llama3LLM instance.
        """
        return {"endpoint": self.endpoint}

# Ensure this class is correctly integrated into the application
if __name__ == "__main__":
    llama_llm = Llama3LLM(endpoint=LLAMA3_API_ENDPOINT, api_key=LLAMA3_API_KEY)
    print(llama_llm._call("Hello, world!"))
