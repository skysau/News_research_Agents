from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import pickle
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

app = FastAPI()


class CombinedRequest(BaseModel):
    urls: Optional[List[str]] = None
    question: str

VECTORSTORE_PATH = "faiss_store_openai.pkl"


@app.post("/process_urls/")
def process_urls(request: CombinedRequest):
    urls = request.urls if request.urls else [os.getenv("DEFAULT_NEWS_URL")]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    return {"message": "URLs processed and vectorstore saved."}


@app.post("/ask/")
def ask_question(request: CombinedRequest):
    if not os.path.exists(VECTORSTORE_PATH):
        raise HTTPException(status_code=404, detail="Vectorstore not found. Please process URLs first.")
    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore_openai = pickle.load(f)
    llm = OpenAI(temperature=0.9, max_tokens=500)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
    result = chain({"question": request.question}, return_only_outputs=True)
    return result


# Combined route: process URLs and answer question in one call
@app.post("/process_and_ask/")
def process_and_ask(request: CombinedRequest):
    urls = request.urls if request.urls else [os.getenv("DEFAULT_NEWS_URL")]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    llm = OpenAI(temperature=0.9, max_tokens=500)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
    result = chain({"question": request.question}, return_only_outputs=True)
    return result
