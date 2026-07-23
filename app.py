import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain.chains import RetrievalQA

load_dotenv()

loader = PyPDFLoader("data/document.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vector_db = FAISS.from_documents(
    docs,
    embeddings
)

retriever = vector_db.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("================================")
print(" RAG Chat With Your Document")
print("================================")

while True:

    question = input("\nAsk a Question: ")

    if question.lower() == "exit":
        break

    answer = qa.run(question)

    print("\nAnswer:\n")
    print(answer)
