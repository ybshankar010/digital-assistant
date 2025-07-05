from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger
from digital_assistant.agents.base_agent import BaseAgent, GraphState

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser

class ChitchatAgent(BaseAgent):

    def __init__(self,db : ChromaDB):
        super().__init__(db)
        self.chitchat_agent_prompt = PromptTemplate.from_template(
            "You are a chitchat agent for IMDB Movie databot. Your task is to respond to the user's query in a friendly manner.\n\n"
            "Previous conversation context:\n{conversation_history}\n\n"
            "Current Query: {query}\n\n"
            "Answer: Understand user query and respond in a friendly manner considering the conversation context. If the query is not related to knowledge, respond with 'I'm just a bot, on top of imdb movie data, your answer is beyond my scope!'\n\n"
        )
        
        self.chitchat_chain = self.chitchat_agent_prompt | self.llm

            

    def invoke(self, state : GraphState):
        """
        Handle chitchat queries.
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = state.get("conversation_history", self._get_conversation_history())
        
        self.logger.debug(f"Chitchat Agent received query: {query}")
        response = self.chitchat_chain.invoke({
            'query': query,
            'conversation_history': conversation_history
        })
        
        answer = response.content.strip()
        self.logger.debug(f"Chitchat Agent response: {answer}")
        
        # Add to memory
        self._add_to_memory(query, answer)
        
        return {
            "input": query, 
            "source": "chitchat", 
            "answer": answer,
            "conversation_history": conversation_history
        }