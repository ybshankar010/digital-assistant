from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.logs.logger import SimpleLogger

from langchain_core.prompts import PromptTemplate

from digital_assistant.utils.common_utils import clean_text
from digital_assistant.agents.base_agent import BaseAgent, GraphState

class IntakeAgent(BaseAgent):

    def __init__(self,db : ChromaDB):
        super().__init__(db)

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
        
    def _extract_metadata(self, query: str) -> dict:
        """
        Extract metadata filters from the user's query using the metadata extraction chain.
        
        Args:
            query (str): The user's query to extract metadata from.

        Returns:
            dict: A dictionary containing the extracted metadata filters.
        """
        try:
            metadata_response = self.metadata_extraction_chain.invoke({"query": query})
            self.logger.debug(f"Extracted metadata response: {metadata_response}")
            metadata_filter = clean_text(metadata_response) if metadata_response else {}
            self.logger.debug(f"Extracted metadata filter: {metadata_filter}")
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            metadata_filter = {}

    def _execute_intake_agent(self, query: str, document_list: list, metadata_list :list,conversation_history: str) -> dict:
        """
        Executes the intake agent with the provided query and document list.
        
        Args:
            query (str): The user's query to process.
            document_list (list): List of documents retrieved from the document store.
            metadata_list (list): List of metadata associated with the documents.
            conversation_history (str): The conversation history to provide context.
            
        Returns:
            dict: A dictionary containing the input query, answer from the intake agent,
                    source of the answer, and updated conversation history.
        """
        
        self.logger.debug(f"meta data :: {metadata_list}")
        titles = ", ".join(metadata["Series_Title"] for metadata in metadata_list[0])
        self.logger.debug(f"Titles :: {titles}")
        flattened_documents = "\n\n".join(doc[0] if isinstance(doc, list) else doc for doc in document_list)
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
            

    def invoke(self, state : GraphState):
        """
        Executes the intake agent with the document store query.
        
        Args:
            state (dict): The state of the knowledge agent.
        """
        query = state["input"]
        conversation_history = state.get("conversation_history", self._get_conversation_history())
        
        self.logger.debug(f"Query to Intake Agent: {query}")
        # Step 1: Extract filters
        metadata_filter = self._extract_metadata(query)
            
        # Step 2: Enhanced query for embedding + filters for metadata
        enhanced_query = f"{conversation_history}\n\nCurrent question: {query}" if conversation_history != "No previous conversation." else query
        self.logger.debug(enhanced_query)
        results = self.document_store.query_data(query=enhanced_query, n_results=3, metadata_filter=metadata_filter)
        
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [{}])
        self.logger.debug(f"Documents retrieved: {len(documents)} items")
        self.logger.debug(f"Total document size: {len(str(documents))} characters")
        
        if documents and documents[0]:
            return self._execute_intake_agent(query, documents, metadatas, conversation_history)

        return {
            "input": query, 
            "source": "forward_to_knowledge",
            "conversation_history": conversation_history
        }