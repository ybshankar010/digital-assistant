from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, START, END

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from typing import TypedDict, Optional, List
from ..utils import clean_text

class GraphState(TypedDict):
    input: str
    source: Optional[str]
    intent: Optional[str]
    answer: Optional[str]
    conversation_history: Optional[str]

class AgenticRAG:

    def __init__(self, db: ChromaDB, max_token_limit: int = 2000):
        self.document_store = db
        self.llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
        self.logger = SimpleLogger(self.__class__.__name__, level="debug")
        
        # Initialize memory with token buffer
        self.memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            return_messages=True
        )
        
        self.intent_classifier_prompt = PromptTemplate.from_template(
            "You are an intent classifier. Your task is to classify the user's query into one of the following intents: chitchat, knowledge.\n\n"
            "Previous conversation context:\n{conversation_history}\n\n"
            "Current Query: {query}\n\n"
            "Classify the intent as either 'chitchat' or 'knowledge'. If the intent is not clear, return 'chitchat'."
        )
        self.intent_classifier_chain = self.intent_classifier_prompt | self.llm

        self.chitchat_agent_prompt = PromptTemplate.from_template(
            "You are a chitchat agent for IMDB Movie databot. Your task is to respond to the user's query in a friendly manner.\n\n"
            "Previous conversation context:\n{conversation_history}\n\n"
            "Current Query: {query}\n\n"
            "Answer: Understand user query and respond in a friendly manner considering the conversation context. If the query is not related to knowledge, respond with 'I'm just a bot, on top of imdb movie data, your answer is beyond my scope!'\n\n"
        )
        self.chitchat_chain = self.chitchat_agent_prompt | self.llm

        self.metadata_extraction_prompt = PromptTemplate.from_template(
            "Extract structured filter conditions from the user's query. "
            "Output only valid JSON using these fields if present: Genre, Released_Year, IMDB_Rating, Director, Star1, Certificate, Series_Title. "
            "If there is only one condition, output it directly as a JSON object. "
            "If there are two or more conditions, wrap them inside an \"$and\" array. "
            "Use operators like \"$gte\" or \"$lte\" where appropriate.\n\n"
            "Query: {query}\n\n"
            "Example (single condition):\n"
            "{{\"Series_Title\": \"21 Grams\"}}\n\n"
            "Example (multiple conditions):\n"
            "{{\n"
            "  \"$and\": [\n"
            "    {{\"Genre\": \"Thriller\"}},\n"
            "    {{\"IMDB_Rating\": {{\"$gte\": 8}}}}\n"
            "  ]\n"
            "}}\n\n"
            "Output:"
        )

        self.metadata_extraction_chain = self.metadata_extraction_prompt | self.llm

        self.intake_agent_prompt = PromptTemplate.from_template(
            "You are an intake agent. Your task is to process the user's query and return relevant information from the document store.\n\n"
            "Previous conversation context:\n{conversation_history}\n\n"
            "Retrieved movie summaries:\n{context}\n\n"
            "Retrieved movie titles:\n{titles}\n\n"
            "Current Query: {query}\n\n"
            "Instructions: Summarize the information from the retrieved chunks and provide a concise, direct answer to the user considering the conversation context. If no relevant documents are found, reply with 'No relevant information found.'\n\n"
            "Answer:"
        )
        self.intake_chain = self.intake_agent_prompt | self.llm
        
        self.knowledge_agent_prompt = PromptTemplate.from_template(
            "You are a knowledge agent. Your task is to answer the user's query using the provided context.\n\n"
            "Previous conversation context:\n{conversation_history}\n\n"
            "Context:\n{context}\n\n"
            "Current Query: {query}\n\n"
            "Answer:"
        )
        self.knowledge_chain = self.knowledge_agent_prompt | self.llm

        self.graph = self._build_graph()

    def _get_conversation_history(self) -> str:
        """
        Get the conversation history from memory as a formatted string.
        """
        return "No previous conversation."  # Default message if no history exists
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

    def _intent_classifier(self, state):
        """
        Classify the intent of the user's query.
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = self._get_conversation_history()
        
        self.logger.debug(f"Classifying intent for query: {query}")
        intent = self.intent_classifier_chain.invoke({
            'query': query,
            'conversation_history': conversation_history
        })
        self.logger.debug(f"Classified intent: {intent}")
        
        return {
            "input": query, 
            'intent': intent.strip(),
            'conversation_history': conversation_history
        }
    
    def _chitchat_agent(self, state):
        """
        Handle chitchat queries.
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = state.get("conversation_history", self._get_conversation_history())
        
        self.logger.debug(f"Chitchat Agent received query: {query}")
        answer = self.chitchat_chain.invoke({
            'query': query,
            'conversation_history': conversation_history
        })
        
        # Add to memory
        self._add_to_memory(query, answer)
        
        return {
            "input": query, 
            "source": "chitchat", 
            "answer": answer,
            "conversation_history": conversation_history
        }

    def _intake_agent(self, state):
        """
        Initialize the intake agent with the document store query.
        
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = state.get("conversation_history", self._get_conversation_history())
        
        self.logger.debug(f"Query to Intake Agent: {query}")
        # Step 1: Extract filters
        try:
            metadata_response = self.metadata_extraction_chain.invoke({"query": query})
            self.logger.debug(f"Extracted metadata response: {metadata_response}")
            metadata_filter = clean_text(metadata_response) if metadata_response else {}
            self.logger.debug(f"Extracted metadata filter: {metadata_filter}")
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            metadata_filter = {}
            
        # Step 2: Enhanced query for embedding + filters for metadata
        enhanced_query = f"{conversation_history}\n\nCurrent question: {query}" if conversation_history != "No previous conversation." else query
        self.logger.debug(enhanced_query)
        results = self.document_store.query_data(query=enhanced_query, n_results=3, metadata_filter=metadata_filter)
        
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [{}])
        self.logger.debug(f"Documents retrieved: {len(documents)} items")
        self.logger.debug(f"Total document size: {len(str(documents))} characters")
        
        if documents and documents[0]:
            self.logger.debug(f"Results found: {results}")
            self.logger.debug(f"meta data :: {metadatas}")
            titles = ", ".join(metadata["Series_Title"] for metadata in metadatas[0])
            self.logger.debug(f"Titles :: {titles}")
            flattened_documents = "\n\n".join(doc[0] if isinstance(doc, list) else doc for doc in documents)
            context_str = flattened_documents[:2000]
            self.logger.debug(f"Context for intake agent: {context_str[:500]}...")

            answer = self.intake_chain.invoke({
                'context': context_str,
                'titles' : titles,
                'query': query,
                'conversation_history': conversation_history
            })
            
            self.logger.debug(f"Intake Agent answer: {answer}")

            # Add to memory
            self._add_to_memory(query, answer)
            
            return {
                "input": query,
                "answer": answer, 
                "source": "intake",
                "conversation_history": conversation_history
            }

        return {
            "input": query, 
            "source": "forward_to_knowledge",
            "conversation_history": conversation_history
        }

    def _knowledge_agent(self, state):
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

        graph.add_node('intent_classifier', self._intent_classifier)
        graph.add_node('chitchat_agent', self._chitchat_agent)
        graph.add_node('intake_agent', self._intake_agent)
        graph.add_node('knowledge_agent', self._knowledge_agent)
        
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

if __name__ == "__main__":
    db = ChromaDB()
    rag = AgenticRAG(db, max_token_limit=1500)
    
    # Test conversation with memory
    queries = [
        "Hi, I'm interested in machine learning",
        "What are different algorithms in sparse graph decomposition?",
        "Can you tell me more about the first algorithm you mentioned?",
        "Thanks for the help!"
    ]
    
    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: {query} ---")
        result = rag.run(query)
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Source: {result.get('source', 'Unknown')}")
        
        # Show memory summary
        memory_summary = rag.get_memory_summary()
        print(f"Memory: {memory_summary['message_count']} messages stored")
    
    # Clear memory example
    print("\n--- Clearing Memory ---")
    rag.clear_memory()