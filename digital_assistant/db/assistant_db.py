import chromadb
from chromadb.utils import embedding_functions
from .base import BaseAssistantDB
from ..utils.constants import *

from pathlib import Path

class ChromaDB(BaseAssistantDB):
    """
    ChromaDB implementation of the Assistant to interact. This class provides methods
    to insert and query data in a ChromaDB database.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChromaDB, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_function=None):
        # Prevent reinitialization
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        self.client = chromadb.PersistentClient(
            path=CHROMADB_PATH
        )
        self.collection = self.get_collection(collection_name=COLLECTION_NAME)

    def insert_data(self, content: str, metadata: dict):
        try:
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[str(hash(content))],
            )
            print(self.collection.get()['ids'])
        except Exception as e:
            print(f"Error inserting data: {e}")

        return
    
    def query_data(self, query: str, n_results: int):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        return results
    
    def get_collection(self, collection_name: str):
        """
        Get a collection from the ChromaDB database.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            Collection: The requested collection.
        """
        return self.client.get_or_create_collection(name=collection_name,
                                                    embedding_function=self.embedding_function)
    
    def get_all_data(self):
        """
        Retrieve all data from the collection.
        This is for Testing purposes and should not be used in production
        as it may return a large amount of data.
        It is recommended to use query_data with specific queries for production use.

        Returns:
            list: A list of all documents in the collection.
        """
        results = self.collection.get(include=["documents", "metadatas", "embeddings"])
        print(f"{self.collection.name} collection has {len(results['documents'])} documents.")
        return results["ids"],results["documents"], results["metadatas"],results["embeddings"]
    
    def delete_data(self, ids: list):
        """
        Delete data from the collection by ids.
        
        Args:
            ids (list): A list of ids to delete from the collection.
        """
        if not ids:
            print("No ids provided for deletion.")
            return
        self.collection.delete(ids=ids)
        return
    
    def init_database(self):
        """
        Initialize the ChromaDB database.
        This method is called to ensure the database is ready for use.
        """
        self.delete_data(ids=self.collection.get()['ids'])
        return

def test():
    client = ChromaDB()
    client.insert_data(
        content="Hello world",
        metadata={"source": "test"}
    )
    client.insert_data(
        content="Goodbye world",
        metadata={"source": "test"},
    )
    results = client.query_data(
        query="Hello world",
        n_results=2,
    )
    print(results)

    return

if __name__ == "__main__":
    test()