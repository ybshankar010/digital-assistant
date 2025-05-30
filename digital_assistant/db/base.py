from abc import ABC, abstractmethod

class BaseAssistantDB(ABC):
    """
    Base class for the AssistantDB. This class defines the  interface for
    inserting and querying data in the database. It should be inherited by
    any specific database implementation (e.g., ChromaDB, SQLite, etc.).
    """
    
    @abstractmethod
    def insert_data(self, content: str, metadata: dict):
        """
        Insert data into the ChromaDB database.

        Args:
            content (str): The content to be inserted.
            metadata (dict): Metadata associated with the content.
        """
        pass

    @abstractmethod
    def query_data(self, query: str, n_results: int):
        """
        Query data from the ChromaDB database.

        Args:
            query (str): The query string.
            n_results (int): The number of results to return.

        Returns:
            list: A list of results matching the query.
        """
        pass