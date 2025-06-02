import pandas as pd
from io import StringIO

from ..utils.constants import DATA_FOLDER
from ..db.assistant_db import ChromaDB

class DataTransformer:
    """
    DataTransformer class to handle the transformation of data before inserting it into the database.
    This class is responsible for transforming raw data into a format suitable for storage in ChromaDB.
    """

    def __init__(self, db: ChromaDB,file_path: str = DATA_FOLDER):
        self.db = db
        self.db.init_database()
        self.file_path = file_path

    def load_data_into_db(self):
        print(f"Indexing data from {self.file_path} into the vector db.")

        df = pd.read_csv(self.file_path, low_memory=False)

        total_records = len(df)
        self._transform_data(df)

        print(f"Finished indexing {total_records} total lines from {self.file_path}.")

        return total_records

    
    def _transform_data(self, data):
        numeric_fields = ["Released_Year", "IMDB_Rating", "Meta_score", "No_of_Votes", "Gross"]

        # Clean up specific fields
        for field in numeric_fields:
            if field in data.columns:
                data[field] = pd.to_numeric(data[field], errors='coerce')  # Coerce invalid to NaN

        for _, row in data.iterrows():
            content = row.get('Overview', '')
            if not content:
                continue

            # Prepare typed metadata
            metadata = {}
            for col, val in row.items():
                if col == "Overview" or col == "Poster_Link":
                    continue
                if pd.notna(val):
                    if col in numeric_fields:
                        try:
                            metadata[col] = float(val) if '.' in str(val) else int(val)
                        except:
                            continue  # Skip invalid values
                    else:
                        metadata[col] = str(val).strip()

            self.db.insert_data(content=content, metadata=metadata)

        print("Data transformation and insertion complete.")
        return data


def index_data():
    """
    Function to index data into the ChromaDB.
    This function is used to load and transform data before inserting it into the database.
    """
    db = ChromaDB()
    transformer = DataTransformer(db)
    transformer.load_data_into_db()
    ids,documents,metadatas,embeddings = db.get_all_data()
    print(f"Total documents in the database: {len(documents)}")
    for id,document,_,_ in zip(ids,documents,metadatas,embeddings):
        print("==========",id, "==========",document[:10])


if __name__ == "__main__":
    index_data()
