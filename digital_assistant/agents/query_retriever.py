from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger
from langgraph.graph import StateGraph, START, END

from digital_assistant.agents.impl.intake_agent import IntakeAgent
from digital_assistant.agents.impl.knowledge_agent import KnowledgeAgent
from digital_assistant.agents.impl.intent_classifier import IntentClassifierAgent
from digital_assistant.agents.impl.chitchat_agent import ChitchatAgent
from digital_assistant.agents.base_agent import GraphState


class AgenticRAG:

    def __init__(self, db: ChromaDB):
        self.document_store = db
        self.logger = SimpleLogger(self.__class__.__name__, level="debug")
        
        self._intent_classifier = IntentClassifierAgent(db)
        self._chitchat_agent = ChitchatAgent(db)
        self._intake_agent = IntakeAgent(db)
        self._knowledge_agent = KnowledgeAgent(db)

        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build the state graph for the query retriever.
        This graph defines the flow of the query through different agents.
        """
        def route_from_intake(state):
            next_step = "knowledge_agent" if state.get("source") == "forward_to_knowledge" else END
            self.logger.debug(f"Routing to: {next_step}")
            return next_step
        
        def route_intent(state):
            next_step = state.get("intent", "").strip()
            self.logger.debug(f"Intent of the query: {next_step}")
            return next_step

        graph = StateGraph(GraphState)

        graph.add_node('intent_classifier', self._intent_classifier.invoke)
        graph.add_node('chitchat_agent', self._chitchat_agent.invoke)
        graph.add_node('intake_agent', self._intake_agent.invoke)
        graph.add_node('knowledge_agent', self._knowledge_agent.invoke)
        
        graph.add_edge(START, 'intent_classifier')

        graph.add_conditional_edges(
            "intent_classifier",
            route_intent,
            {
                "chitchat": "chitchat_agent",
                "knowledge": "intake_agent"
            }
        )
        
        graph.add_conditional_edges(
            'intake_agent',
            route_from_intake,
            {
                'knowledge_agent': 'knowledge_agent',
                END: END
            }
        )
        
        graph.add_edge('chitchat_agent', END)
        graph.add_edge('knowledge_agent', END)

        return graph.compile()
    
    def run(self, query):
        """
        Run the query retriever with the provided query.
        
        Args:
            query (str): The query to be processed by the agents.
        
        Returns:
            dict: The final state after processing the query.
        """
        initial_state = {'input': query}
        final_state = self.graph.invoke(initial_state)
        return final_state

if __name__ == "__main__":
    db = ChromaDB()
    rag = AgenticRAG(db)
    
    queries = [
        "Hi, I'm interested in machine learning",
        "Okay can you talk about movie bahubali",
        "What is the capital of India?",
        "Can you tell me about the movie Inception?",
        "What is the weather like today?",
        "Thanks for the help!"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: {query} ---")
        result = rag.run(query)
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Source: {result.get('source', 'Unknown')}")