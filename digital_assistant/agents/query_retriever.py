from ..db.assistant_db import ChromaDB
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from typing import TypedDict, Optional

class GraphState(TypedDict):
    input: str
    source: Optional[str]
    answer: Optional[str]



class AgenticRAG:

    def __init__(self,db: ChromaDB):
        self.document_store = db
        self.llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
        self.knowledge_agent_prompt = PromptTemplate.from_template(
            "You are a knowledge agent. Your task is to answer the user's query using the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Query: {query}\n\n"
            "Answer:"
        )
        self.chain = self.knowledge_agent_prompt | self.llm

        self.graph = self._build_graph()



    def _intake_agent(self, state):
        query = state["input"]
        print(f"Query to Intake Agent: {query}")
        
        results = self.document_store.query_data(query=query, n_results=3)
        print(f"Results from Vector db: {results}")
        
        documents = results.get("documents", [[]])[0]
        if documents:
            answer = " ".join(documents)
            print(f"Answer from Intake Agent: {answer}")
            return {"answer": answer, "source": "intake"}

        return {"input": query, "source": "forward_to_knowledge"}

    

    def _knowledge_agent(self, state):
        """
        Initialize the knowledge agent with the duckduckgo search tool.
        
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        print(f"Query to Knowledge Agent: {query}")
        
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            snippets = [result['body'] for result in results]
        
        answer = self.chain.invoke({'context':snippets, 'query':query})
        return {'input':query, 'source':'knowledge', 'answer':answer}
    

    def _build_graph(self):
        """
        Build the state graph for the query retriever.
        This graph defines the flow of the query through different agents.
        """
        def route_from_intake(state):
            if state.get("source") == "forward_to_knowledge":
                return "knowledge_agent"
            return "end" 
        graph = StateGraph(GraphState)

        graph.add_node('intake_agent', self._intake_agent)
        graph.add_node('knowledge_agent', self._knowledge_agent)
        
        graph.add_conditional_edges(
            'intake_agent',
            route_from_intake,
            {'knowledge_agent':'knowledge_agent'}
        )
        
        graph.set_entry_point('intake_agent')
        graph.set_finish_point('knowledge_agent')
        graph.set_finish_point('intake_agent')

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
    rag = AgenticRAG(ChromaDB())
    query = "What is the capital of France?"
    result = rag.run(query)
    print(result)