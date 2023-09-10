# For document loading
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader


def docs_from_src(_path, src_type):
    if src_type == 'local':
        loader = DirectoryLoader(_path)
    elif src_type == 'pdf':
        loader = PyPDFLoader(_path)
    elif src_type == 'web':
        loader = WebBaseLoader(_path)
    docs = loader.load()

    return docs


if __name__ == '__main__':
    docs = docs_from_src('docs/CS229_Lecture_Notes.pdf', src_type='pdf')
    print(len(docs))
    print(docs[0].page_content[0:500])
