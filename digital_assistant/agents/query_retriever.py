from ..db.assistant_db import ChromaDB
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, START, END

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate

from typing import TypedDict, Optional

class GraphState(TypedDict):
    input: str
    source: Optional[str]
    answer: Optional[str]

class AgenticRAG:

    def __init__(self,db: ChromaDB):
        self.document_store = db
        self.llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
        self.intake_agent_prompt = PromptTemplate.from_template(
            "You are an intake agent. Your task is to process the user's query and return relevant information from the document store.\n\n"
            "Retrieved chunks:\n{context}\n\n"
            "Query: {query}\n\n"
            "Answer: If you find relevant documents, return them as a single string."
        )
        self.intake_chain = self.intake_agent_prompt | self.llm
        
        self.knowledge_agent_prompt = PromptTemplate.from_template(
            "You are a knowledge agent. Your task is to answer the user's query using the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Query: {query}\n\n"
            "Answer:"
        )
        self.knowledge_chain = self.knowledge_agent_prompt | self.llm

        self.graph = self._build_graph()



    def _intake_agent(self, state):
        """
        Initialize the intake agent with the duckduckgo search tool.
        
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        print(f"Query to Intake Agent: {query}")
        
        results = self.document_store.query_data(query=query, n_results=3)
        
        documents = results.get("documents", [[]])[0]
        if documents:
            answer = self.intake_chain.invoke({'context': documents, 'query': query})
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
        
        print(f"Snippets from DuckDuckGo: {snippets}")
        answer = self.knowledge_chain.invoke({'context':snippets, 'query':query})
        return {'input':query, 'source':'knowledge', 'answer':answer}

    def _build_graph(self):
        """
        Build the state graph for the query retriever.
        This graph defines the flow of the query through different agents.
        """
        def route_from_intake(state):
            next_step = "knowledge_agent" if state.get("source") == "forward_to_knowledge" else END
            print("Routing to:", next_step)
            return next_step

        graph = StateGraph(GraphState)

        graph.add_node('intake_agent', self._intake_agent)
        graph.add_node('knowledge_agent', self._knowledge_agent)
        
        graph.add_edge(START, 'intake_agent')
        
        graph.add_conditional_edges(
            'intake_agent',
            route_from_intake,
            {'knowledge_agent':'knowledge_agent',
             END: END}
        )

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
    # query = "What is the capital of France?"
    query = "what are different algorithms in sparse graph decomposition?"
    result = rag.run(query)
    print(result)