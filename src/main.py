import os
import sys
import io
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import streamlit as st
import logging
import plotly.express as px
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure Blob Storage settings
AZURE_BLOB_ACCOUNT_URL = os.getenv('AZURE_BLOB_ACCOUNT_URL')
AZURE_BLOB_CONTAINER_NAME = os.getenv('AZURE_BLOB_CONTAINER_NAME')
LLAMA3_API_ENDPOINT = os.getenv('LLAMA3_API_ENDPOINT')
LLAMA3_API_KEY = os.getenv('LLAMA3_API_KEY')

# Verify environment variables are loaded
logger.info(f"AZURE_BLOB_ACCOUNT_URL: {AZURE_BLOB_ACCOUNT_URL}")
logger.info(f"AZURE_BLOB_CONTAINER_NAME: {AZURE_BLOB_CONTAINER_NAME}")
logger.info(f"LLAMA3_API_ENDPOINT: {LLAMA3_API_ENDPOINT}")

# Ensure required environment variables are set
if not all([AZURE_BLOB_ACCOUNT_URL, AZURE_BLOB_CONTAINER_NAME, LLAMA3_API_ENDPOINT, LLAMA3_API_KEY]):
    raise ValueError("One or more required environment variables are not set.")

# Add the 'src' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.file_handler import FileHandler
from src.data_processor import DataProcessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.summarizer import Summarizer
from src.llama3_llm import Llama3LLM

# Initialize the Llama3LLM and Summarizer classes
llm = Llama3LLM(endpoint=LLAMA3_API_ENDPOINT, api_key=LLAMA3_API_KEY)
summarizer = Summarizer(llm)

@st.cache_data
def load_and_process_data(file_name: str) -> DataProcessor:
    """
    Load and process data from an Azure Blob Storage file.
    
    Parameters:
    file_name (str): The name of the file to load and process.
    
    Returns:
    DataProcessor: An instance of DataProcessor with the loaded data.
    """
    try:
        logger.debug(f"Loading and processing data from file: {file_name}")
        blob_service_client = BlobServiceClient(account_url=AZURE_BLOB_ACCOUNT_URL, credential=DefaultAzureCredential())
        container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(file_name)
        
        # Download the blob content
        blob_data = blob_client.download_blob().readall()
        logger.debug("Blob data downloaded successfully")
        
        # Convert bytes to a file-like object
        file_like_object = io.BytesIO(blob_data)
        
        # Read the CSV file
        data = pd.read_csv(file_like_object)
        logger.debug(f"CSV file read successfully. Shape: {data.shape}")
        return DataProcessor(data)
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {str(e)}", exc_info=True)
        raise

@st.cache_data
def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the provided text.
    
    Parameters:
    text (str): The text to analyze.
    
    Returns:
    str: The sentiment analysis result.
    """
    logger.debug("Analyzing sentiment")
    analyzer = SentimentAnalyzer(llm)
    return analyzer.analyze(text)

@st.cache_data
def summarize_data(_processor: DataProcessor, group_or_entity: str, name: str) -> str:
    """
    Generate a summary for the specified group or entity using the data processor.
    
    Parameters:
    _processor (DataProcessor): An instance of the DataProcessor class to generate the summary.
    group_or_entity (str): The type of summary to generate (e.g., group or entity).
    name (str): The name of the group or entity.
    
    Returns:
    str: The generated summary.
    """
    logger.debug(f"Summarizing data for {group_or_entity}: {name}")
    return summarizer.summarize(_processor, group_or_entity, name)

def main():
    """
    Main function to run the Streamlit application for assessment data analysis.
    """
    st.title("Assessment Data Analyzer")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            logger.debug("File uploaded. Starting processing.")
            
            # Upload the file to blob storage
            file_name = FileHandler.upload_to_blob_storage(uploaded_file)
            logger.debug(f"File uploaded to blob storage: {file_name}")
            
            # Process the uploaded file
            logger.debug("Starting to process the uploaded file")
            processor = load_and_process_data(file_name)
            logger.debug("File processed successfully")

            st.sidebar.header("Select Analysis Type")
            analysis_type = st.sidebar.radio("Choose analysis type:", ("Entity", "Group"))
            logger.debug(f"Analysis type selected: {analysis_type}")

            if analysis_type == "Entity":
                options = processor.get_entities()
            else:
                options = processor.get_groups()
            logger.debug(f"Options retrieved: {options}")

            selected = st.sidebar.selectbox(f"Select a {analysis_type.lower()}:", options)
            logger.debug(f"Selected {analysis_type.lower()}: {selected}")

            st.subheader(f"Assessment Data for {selected}")
            assessment_data = processor.get_assessment_data(selected, is_group=(analysis_type == "Group"))
            logger.debug(f"Assessment data retrieved. Type: {type(assessment_data)}")
            logger.debug(f"First few rows of assessment data: {assessment_data[:5] if isinstance(assessment_data, list) else assessment_data.head()}")
            st.dataframe(assessment_data)

            progress = processor.get_progress(selected, is_group=(analysis_type == "Group"))
            logger.debug("Progress data retrieved")
            st.subheader("Progress Over Time")
            fig = px.line(x=progress.index, y=progress.values, labels={'x': 'Assessment Number', 'y': 'Average Rating'})
            st.plotly_chart(fig)

            capability_scores = processor.get_capability_scores(selected, is_group=(analysis_type == "Group"))
            logger.debug("Capability scores retrieved")
            st.subheader("Capability Scores")
            fig = px.bar(x=list(capability_scores.keys()), y=list(capability_scores.values()), labels={'x': 'Capability', 'y': 'Average Score'})
            st.plotly_chart(fig)

            criteria_distribution = processor.get_criteria_distribution(selected, is_group=(analysis_type == "Group"))
            logger.debug("Criteria distribution retrieved")
            st.subheader("Criteria Stage Distribution")
            fig = px.pie(values=list(criteria_distribution.values()), names=list(criteria_distribution.keys()))
            st.plotly_chart(fig)

            st.subheader("Comprehensive Analysis")
            logger.debug("Starting comprehensive analysis")
            summary = summarize_data(processor, analysis_type, selected)
            logger.debug("Comprehensive analysis completed")
            st.write(summary)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
