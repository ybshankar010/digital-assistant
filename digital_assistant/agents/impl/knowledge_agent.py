from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger
from digital_assistant.agents.base_agent import BaseAgent, GraphState

from langchain_core.prompts import PromptTemplate
from duckduckgo_search import DDGS

class KnowledgeAgent(BaseAgent):

    def __init__(self,db : ChromaDB):
        super().__init__(db)
        
        self.knowledge_agent_prompt = PromptTemplate.from_template(
            "You are a knowledge agent. Your task is to answer the user's query using the provided context.\n\n"
            "Previous conversation context:\n{conversation_history}\n\n"
            "Context:\n{context}\n\n"
            "Current Query: {query}\n\n"
            "Answer:"
        )
        self.knowledge_chain = self.knowledge_agent_prompt | self.llm

            

    def invoke(self, state : GraphState):
        """
        Initialize the knowledge agent with the duckduckgo search tool.
        
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = state.get("conversation_history", self._get_conversation_history())
        
        self.logger.debug(f"Query to Knowledge Agent: {query}")
        
        # Enhance query with conversation context for better search
        enhanced_query = f"{query} {conversation_history}" if conversation_history != "No previous conversation." else query
        
        with DDGS() as ddgs:
            results = ddgs.text(enhanced_query[:500], max_results=5)  # Limit query length
            snippets = [result['body'] for result in results]
        
        self.logger.info(f"Executing duckduckgo search for query: {query}")
        self.logger.debug(f"Snippets from DuckDuckGo: {snippets}")
        answer = self.knowledge_chain.invoke({
            'context': snippets, 
            'query': query,
            'conversation_history': conversation_history
        })
        
        # Add to memory
        self._add_to_memory(query, answer)
        
        return {
            'input': query, 
            'source': 'knowledge', 
            'answer': answer,
            'conversation_history': conversation_history
        }