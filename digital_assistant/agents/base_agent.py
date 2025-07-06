from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationTokenBufferMemory

from typing import TypedDict, Optional
from digital_assistant.utils import setup_llm
from abc import ABC, abstractmethod

class GraphState(TypedDict):
    input: str
    source: Optional[str]
    intent: Optional[str]
    answer: Optional[str]
    conversation_history: Optional[str]

class BaseAgent(ABC):
    """
    Base class for all agents in the digital assistant system.
    This class defines the interface for running queries and managing memory.
    """

    def __init__(self, db: ChromaDB, max_token_limit: int = 2000):
        self.document_store = db
        self.logger = SimpleLogger(self.__class__.__name__, level="debug")

        self.schema = {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "answer":  {"type": "string"}
            },
            "required": ["thought", "answer"]
        }

        self.output_parser = JsonOutputParser(json_schema=self.schema)
        self.llm = setup_llm("llama3.1:8b", temperature=0.1)

        # Initialize memory with token buffer
        self.memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            return_messages=True
        )
        

    def _get_conversation_history(self) -> str:
        """
        Get the conversation history from memory as a formatted string.
        """
        return "No previous conversation."
        # try:
        #     messages = self.memory.chat_memory.messages
        #     if not messages:
        #         return "No previous conversation."
            
        #     history_parts = []
        #     for message in messages:
        #         if isinstance(message, HumanMessage):
        #             history_parts.append(f"Human: {message.content}")
        #         elif isinstance(message, AIMessage):
        #             history_parts.append(f"Assistant: {message.content}")
            
        #     return "\n".join(history_parts[-6:])  # Keep last 6 messages for context
        # except Exception as e:
        #     print(f"Error retrieving conversation history: {e}")
        #     return "No previous conversation."

    def _add_to_memory(self, human_input: str, ai_response: str):
        """
        Add human input and AI response to memory.
        """
        try:
            self.memory.chat_memory.add_user_message(human_input)
            self.memory.chat_memory.add_ai_message(ai_response)
        except Exception as e:
            self.logger.error(f"Error adding to memory: {e}")
    
    def clear_memory(self):
        """
        Clear the conversation memory.
        """
        self.memory.clear()
        self.logger.debug("Conversation memory cleared.")
        return
    
    def get_memory_summary(self):
        """
        Get a summary of the current memory state.
        """
        try:
            messages = self.memory.chat_memory.messages
            return {
                "message_count": len(messages),
                "memory_content": self._get_conversation_history()
            }
        except Exception as e:
            return {"error": f"Error getting memory summary: {e}"}
    
    @abstractmethod
    def invoke(self, agentstate: GraphState) -> str:
        """
        Run a query against the agent's knowledge base and return the response.
        
        Args:
            query (str): The query string to process.
        
        Returns:
            str: The response from the agent.
        """
        pass
