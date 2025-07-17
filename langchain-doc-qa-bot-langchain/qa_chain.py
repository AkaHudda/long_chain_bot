from langchain_groq import ChatGroq
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vector_store")
    return db

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vector_store", embeddings)

def create_qa_chain(vector_store):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",  # ✅ Use this model instead
        temperature=0.3
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
