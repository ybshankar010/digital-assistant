import pandas as pd
from ..utils.constants import DATA_FOLDER
from ..db.assistant_db import ChromaDB



class DataTransformer:
    """
    DataTransformer class to handle the transformation of data before inserting it into the database.
    This class is responsible for transforming raw data into a format suitable for storage in ChromaDB.
    """

    BATCH_SIZE = 5

    def __init__(self, db: ChromaDB,file_path: str = DATA_FOLDER):
        self.db = db
        self.db.init_database()
        self.file_path = file_path

    def load_data_into_db(self):

        print(f"Indexing data from {self.file_path} into the vector db.")

        if self.BATCH_SIZE != -1:
            with open(self.file_path, 'r') as f:
                lines = [next(f) for _ in range(self.BATCH_SIZE)]
            df = pd.read_json(''.join(lines), lines=True)
            print(f"Loaded {self.BATCH_SIZE} lines from {self.file_path}.")
        else:
            df = pd.read_json(self.file_path, lines=True)
            print(f"Loaded entire file: {self.file_path}.")
        
        self._transform_data(df)
        return
    
    def _transform_data(self, data):
        """
        Transform the data into a format suitable for storage in ChromaDB.
        """
        for _, row in data.iterrows():
            content = row.get('abstract', '')
            metadata = row.to_dict()
            metadata.pop('abstract', None)
            cleaned_meta_data = { k: v for k, v in row.items() if isinstance(v, (str, int, float, bool)) and v is not None}

            # print(f"Processing index : {metadata}")

            if content:
                self.db.insert_data(content=content, metadata=cleaned_meta_data)
                # print(f"Inserted data for index {index}: {content[:50]}")
        print("Data transformation and insertion complete.")
        return data


if __name__ == "__main__":
    db = ChromaDB()
    transformer = DataTransformer(db)
    transformer.load_data_into_db()
    ids,documents,metadatas,embeddings = db.get_all_data()
    print(f"Total documents in the database: {len(documents)}")
    for id,document,metadata,embedding in zip(ids,documents,metadatas,embeddings):  # Print first 5 results
        print("==========",id, "==========",metadata,"==========",document[:10],"==========",embedding[:10])
