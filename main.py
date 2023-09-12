import os

from config import *
from doc_loader import *
from doc_splitter import *
from doc_indexing import *
from doc_qa import *

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

import langchain
from langchain.llms import HuggingFacePipeline, OpenLM

from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()


app = FastAPI()

embedding = HuggingFaceEmbeddings()
vec_storage = VectorDBStorage(embedding, PATH_VECTOR_DB)

#for f in ['CS229_Lecture_Notes.pdf', '10.1.1.693.855.pdf']:
#    docs = docs_from_src(PATH_DOCS+'{}'.format(f), src_type='pdf')
#    splitted_docs = doc_splitting(docs)
#    vec_storage.doc_storing(splitted_docs)

llm_hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 30},
)

#question = 'What is optimization?'



@app.get("/clear_db")
async def clear_db():
    vec_storage.erase_all()

    return {"success": "Clear all data!"}


@app.post("/upload_pdf/")
async def upload_pdf_to_db(File: UploadFile):
    tmp_loc = os.path.join('tmp', File.filename)
    with open(tmp_loc, 'wb+') as file_obj:
        file_obj.write(File.file.read())

    print("load the doc from source")
    docs = docs_from_src(tmp_loc, src_type='pdf')
    print("splitting doc using llm")
    splitted_docs = doc_splitting(docs)
    print("add doc to database")
    vec_storage.doc_storing(splitted_docs)

    return {"success": "Uploaded file {} to vector database".format(File.filename)}


@app.post("/query")
async def query_from_db(query: str):
    ids = vec_storage.get()
    if len(ids) == 0:
        return {"answer": "No data in database!"}

    res = doc_qa(
        llm=llm_hf,
        retriever=vec_storage.vec_db_storage.as_retriever(),
        question=query
    )

    return {"answer": res}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)

#question = 'What is convex optimization?'
#ans = vec_storage.doc_search(question)
#print(ans)
#print('---------')
#print(vec_storage.vec_db_storage.get())

