import re
import json

from langchain_ollama.llms import OllamaLLM

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
            print("Metadata response was not a dictionary.")
            return {}
    except Exception as e:
        print(f"Failed to parse metadata filter: {e}\nRaw response: {raw_response}")
        return {}
    
def setup_llm(model_name: str, temperature: float = 0.1):
    """
    Setup the LLM with the specified model name and temperature.
    
    Args:
        model_name (str): The name of the LLM model to use.
        temperature (float): The temperature setting for the LLM.
        
    Returns:
        OllamaLLM: An instance of the OllamaLLM with the specified settings.
    """
    
    return OllamaLLM(model=model_name, temperature=temperature)