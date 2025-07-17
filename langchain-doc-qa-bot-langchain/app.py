import os
from dotenv import load_dotenv
from utils import load_document, chunk_documents
from qa_chain import build_vector_store, create_qa_chain

load_dotenv()

file_path = "example_docs/sample.txt"  
docs = load_document(file_path)
chunks = chunk_documents(docs)

db = build_vector_store(chunks)
qa = create_qa_chain(db)


print("\nAsk questions about the document (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    result = qa({"query": query})
    print("Bot:", result["result"])
 
