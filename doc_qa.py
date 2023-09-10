from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain.llms import HuggingFacePipeline #, CTransformers


def doc_qa(llm, retriever, question):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever
    )

    res = qa_chain({
        "query": question
    })

    return res["result"]


def doc_qa_with_promt(llm, retriever, question):
    pass