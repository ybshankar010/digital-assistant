import pandas as pd
from io import StringIO

from ..utils.constants import DATA_FOLDER
from ..db.assistant_db import ChromaDB



class DataTransformer:
    """
    DataTransformer class to handle the transformation of data before inserting it into the database.
    This class is responsible for transforming raw data into a format suitable for storage in ChromaDB.
    """

    BATCH_SIZE = -1

    def __init__(self, db: ChromaDB,file_path: str = DATA_FOLDER):
        self.db = db
        self.db.init_database()
        self.file_path = file_path

    # def load_data_into_db(self):
    #     print(f"Indexing data from {self.file_path} into the vector db.")

    #     batch_size = self.BATCH_SIZE if self.BATCH_SIZE > 0 else 1000
    #     total_lines = 0
    #     batch_num = 0

    #     with open(self.file_path, 'r') as f:
    #         while True:
    #             lines = list()
    #             try:
    #                 for _ in range(batch_size):
    #                     lines.append(next(f))
    #             except StopIteration:
    #                 pass  # Reached end of file

    #             if not lines:
    #                 break  # No more data

    #             batch_num += 1
    #             json_str = StringIO(''.join(lines))  # Wrap in StringIO
    #             df_batch = pd.read_json(json_str, lines=True)
    #             print(f"Processing batch {batch_num} with {len(df_batch)} records.")

    #             self._transform_data(df_batch)
    #             total_lines += len(df_batch)

    #     print(f"Finished indexing {total_lines} total lines from {self.file_path}.")

    #     return total_lines

    def load_data_into_db(self):
        print(f"Indexing data from {self.file_path} into the vector db.")

        df = pd.read_csv(self.file_path, low_memory=False)

        total_records = len(df)
        self._transform_data(df)

        print(f"Finished indexing {total_records} total lines from {self.file_path}.")

        return total_records

    
    def _transform_data(self, data):
        """
        Transform the data into a format suitable for storage in ChromaDB.
        """
        for _, row in data.iterrows():
            content = row.get('Overview', '')
            metadata = row.to_dict()
            metadata.pop('Overview', None)
            cleaned_meta_data = { k: v for k, v in row.items() if isinstance(v, (str, int, float, bool)) and v is not None}

            # print(f"Processing index : {metadata}")

            if content:
                self.db.insert_data(content=content, metadata=cleaned_meta_data)
                # print(f"Inserted data for index {index}: {content[:50]}")
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
