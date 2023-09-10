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


#if __name__ == '__main__':
    