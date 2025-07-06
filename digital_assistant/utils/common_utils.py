import re
import json

from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from digital_assistant.logs.logger import SimpleLogger
logger = SimpleLogger(__name__, level="debug")

def clean_text(raw_response: str) -> dict:
    """
    Safely parse the LLM response and return a metadata filter dict.
    Handles ```json blocks and malformed JSON gracefully.
    """
    try:
        # Remove markdown-style code fences (```json ... ```)
        cleaned = re.sub(r"```(?:json)?", "", raw_response, flags=re.IGNORECASE).strip("` \n")

        # Parse JSON
        metadata_filter = json.loads(cleaned)
        if isinstance(metadata_filter, dict):
            return metadata_filter
        else:
            logger.debug("Metadata response was not a dictionary.")
            return {}
    except Exception as e:
        logger.error(f"Failed to parse metadata filter: {e}\nRaw response: {raw_response}")
        return {}
    
def setup_llm(model_name: str = "llama3.1:8b", temperature: float = 0.1):
    """
    Setup the LLM with the specified model name and temperature.
    
    Args:
        model_name (str): The name of the LLM model to use.
        temperature (float): The temperature setting for the LLM.
        
    Returns:
        llm instance with specified settings.
    """
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature
    )
    # llm = OllamaLLM(
    #     model=model_name,
    #     temperature=temperature,
    #     format="json"
    # )
    
    return llm

def get_overview_content(doc: str) -> str:
    """
    Extracts the first 200 characters from the overview content.

    Args:
        doc (str): The overview content from which to extract.
        
    Returns:
        str: The first 200 characters of the overview, or an empty string if the overview is None.
    """
    overview_regex = re.compile(r"overview:\s*", re.IGNORECASE)
    m = overview_regex.search(doc)
    return doc[m.end():].lstrip() if m else doc

if __name__ == "__main__":
    llm = setup_llm()
    logger.info(llm.invoke("Return {\"answer\": \"test\"} as JSON"))