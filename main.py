import os

from config import *
from doc_indexing import *
from doc_qa import *

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

import langchain
from langchain.llms import HuggingFacePipeline, OpenLM

from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()

# Initialization the mandatory objects
# The main app
app = FastAPI()
# Embedding function
embedding = HuggingFaceEmbeddings()
# Vector DB storage
vec_storage = VectorDBStorage(embedding, PATH_VECTOR_DB, "courses")
# LLM model for chat generation
llm_hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",  # If enough resource, one might use bigger models...
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 30, "temperature": 0.2},
)


@app.get("/clear_db")
async def clear_db():
    vec_storage.erase_all()

    return {"success": "Clear all data!"}


@app.post("/upload_pdf/")
async def upload_pdf_to_db(File: UploadFile):
    tmp_loc = os.path.join("tmp", File.filename)
    with open(tmp_loc, "wb+") as file_obj:
        file_obj.write(File.file.read())

    doc = vec_storage.load_doc_pdf(tmp_loc)
    splitted_doc = vec_storage.doc_splitting(doc)
    vec_storage.doc_storing(splitted_doc)

    return {"success": "Uploaded file {} to vector database".format(File.filename)}


@app.post("/upload_docx/")
async def upload_docx_to_db(File: UploadFile):
    tmp_loc = os.path.join("tmp", File.filename)
    with open(tmp_loc, "wb+") as file_obj:
        file_obj.write(File.file.read())

    doc = vec_storage.load_doc_docx(tmp_loc)
    splitted_doc = vec_storage.doc_splitting(doc)
    vec_storage.doc_storing(splitted_doc)

    return {"success": "Uploaded file {} to vector database".format(File.filename)}


@app.post("/query")
async def query_from_db(query: str):
    ids = vec_storage.vec_db_storage.get()
    if len(ids) == 0:
        return {"answer": "No data in database!"}

    res = doc_qa(
        llm=llm_hf, retriever=vec_storage.vec_db_storage.as_retriever(return_source_documents=True), question=query
    )

    return {"answer": res}


@app.get("/listing")
async def get_document_list():
    obj = vec_storage.vec_db_storage.get()

    return obj


@app.post("/search")
async def doc_search(query: str):
    return vec_storage.doc_search(query)


@app.post("/info_extract")
async def info_extract(template: str, query: str):
    output = doc_qa_with_promt(llm_hf, vec_storage, template, query)

    return {'summary': output}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# question = 'What is convex optimization?'
# ans = vec_storage.doc_search(question)
# print(ans)
# print('---------')
# print(vec_storage.vec_db_storage.get())
