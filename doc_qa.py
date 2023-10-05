from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.question_answering import load_qa_chain
from prompt_engineer import *
# from langchain.llms import HuggingFacePipeline #, CTransformers


def doc_qa(llm, retriever, question):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    res = qa_chain({"query": question})

    return res["result"]


def doc_qa_indexing(llm, ):


def doc_qa_with_promt(llm, vectordb, template, query):
    parser = PydanticOutputParser(pydantic_object=RegionOutlookList)
    prompt = create_prompt(template, parser)
    chain = load_qa_chain(llm=llm, chain_type="map_reduce", prompt=prompt)
    docs = vectordb.doc_search(query, k=3)
    res = chain({"input_documents": docs, "question": query})
    
    return res['output_text']


"""
You are a helpful assistant with expertise in trade and international economics
List of all regions and for each region describe in a few sentences the state of the topics of economy and environment
"""