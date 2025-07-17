import os
import streamlit as st
from dotenv import load_dotenv
from utils import load_document, chunk_documents
from qa_chain import build_vector_store, create_qa_chain

load_dotenv()

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    path = f"temp.{uploaded_file.name.split('.')[-1]}"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        docs = load_document(path)
        chunks = chunk_documents(docs)
        db = build_vector_store(chunks)
        qa = create_qa_chain(db)

    st.success("✅ Document processed! Ask your question below.")
    question = st.text_input("🧠 Ask a question:")

    if question:
        response = qa.invoke({"query": question})
        st.markdown("### 🤖 Answer")
        st.write(response["result"])

        with st.expander("📚 Source Documents"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i+1}**: {doc.metadata.get('source', 'Unknown')}")
                st.markdown(doc.page_content[:500] + "...")
