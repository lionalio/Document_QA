from config import *
from doc_loader import *
from doc_splitter import *
from doc_indexing import *
from doc_qa import *
import langchain
from langchain.llms import HuggingFacePipeline, OpenLM

from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()


embedding = HuggingFaceEmbeddings()
vec_storage = VectorDBStorage(embedding, PATH_VECTOR_DB)

for f in ['CS229_Lecture_Notes.pdf', '10.1.1.693.855.pdf']:
    docs = docs_from_src(PATH_DOCS+'{}'.format(f), src_type='pdf')
    splitted_docs = doc_splitting(docs)
    vec_storage.doc_storing(splitted_docs)

llm_hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 30},
)

question = 'What is optimization?'

res = doc_qa(
    llm=llm_hf,
    retriever=vec_storage.vec_db_storage.as_retriever(),
    question=question
)

print(res)

#question = 'What is convex optimization?'
#ans = vec_storage.doc_search(question)
#print(ans)
#print('---------')
#print(vec_storage.vec_db_storage.get())

