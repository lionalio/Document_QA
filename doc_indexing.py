from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma # Pinecone


class VectorDBStorage:
    def __init__(self, embedding, storage_path):
        self.embedding = embedding
        self.storage_path = storage_path
        self.vec_db_storage = None

    def doc_storing(self, splitted_docs):
        self.vec_db_storage = Chroma.from_documents(
            documents=splitted_docs,
            embedding=self.embedding,
            persist_directory = self.storage_path
        )

    def doc_search(self, question, k=3):
        docs = self.vec_db_storage.similarity_search(question, k=k)
        self.vec_db_storage.persist()

        ret = {}
        for i, doc in enumerate(docs):
            ret[i] = doc.page_content

        return ret

    def erase_all(self):
        print('Warning: Deleting the entire database')
        if self.vec_db_storage is None:
            print("No database to delete!")
            return
        
        ids = []
        for id_ in self.vec_db_storage.get():
            ids.append(id_)

        if len(ids) == 0:
            print("Database is empty already")
            return
        else:
            self.vec_db_storage.delete(ids)
            print('Deletion completed!')
            return 

#if __name__ == '__main__':
    