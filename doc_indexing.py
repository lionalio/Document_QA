from langchain.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # Pinecone


class VectorDBStorage:
    def __init__(self, embedding, storage_path, collection_name):
        self.embedding = embedding
        self.storage_path = storage_path
        self.vec_db_storage = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=storage_path,
        )

    def load_doc_pdf(self, doc_path):
        loader = PyPDFLoader(doc_path)
        doc = loader.load()

        return doc

    def load_doc_docx(self, doc_path):
        loader = Docx2txtLoader(doc_path)
        doc = loader.load()

        return doc

    def doc_splitting(self, loaded_doc, chunk_size=1000, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splitted_doc = splitter.split_documents(loaded_doc)

        return splitted_doc

    def doc_storing(self, splitted_doc):
        self.vec_db_storage.add_documents(
            documents=splitted_doc,
            embedding=self.embedding,
            persist_directory=self.storage_path,
        )
        self.vec_db_storage.persist()

    def doc_search(self, question, k=3):
        docs = self.vec_db_storage.similarity_search(question, k=k, include_metadata=True)
        self.vec_db_storage.persist()

        #ret = {}
        #for i, doc in enumerate(docs):
        #    ret[i] = doc.page_content

        return docs

    def erase_all(self):
        print("Warning: Deleting the entire database")
        self.vec_db_storage.delete_collection()
        self.vec_db_storage.persist()
