from langchain.text_splitter import RecursiveCharacterTextSplitter


def doc_splitting(docs, chunk_size=1000, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
        )
    docs_ = splitter.split_documents(docs)

    return docs_


