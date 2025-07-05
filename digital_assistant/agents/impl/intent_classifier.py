from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger
from digital_assistant.agents.base_agent import BaseAgent, GraphState

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class IntentClassifierAgent(BaseAgent):

    def __init__(self,db : ChromaDB):
        super().__init__(db)
        
        self.intent_classifier_prompt = PromptTemplate.from_template(
             "You are an intent classifier. Your task is to classify the user's query into one of the following intents: chitchat, knowledge.\n\n"
             "Knowledge intent refers to queries that seek factual information, about movies topic alone. Imdb data is indexed and can be answered when knowledge intent is choosen.\n"
             "Previous conversation context:\n{conversation_history}\n\n"
             "Current Query: {query}\n\n"
             "Classify the intent as either 'chitchat' or 'knowledge'. If the intent is not clear, return 'chitchat'."
             "Just return the intent as a single word without any additional text.\n\n"
            "Answer:"
        )

        self.intent_classifier_chain = (
            self.intent_classifier_prompt
            | self.llm
        )

            

    def invoke(self, state : GraphState):
        """
        Classify the intent of the user's query.
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = self._get_conversation_history()
        
        self.logger.debug(f"Classifying intent for query: {query}")
        response = self.intent_classifier_chain.invoke({
            'query': query,
            'conversation_history': conversation_history
        })
        intent = response.content.strip()
        if not intent:
            intent = "chitchat"
        intent = intent.lower()
        self.logger.debug(f"Classified intent: {intent}")

        return {
            "input": query, 
            'intent': intent,
            'conversation_history': conversation_history
        }