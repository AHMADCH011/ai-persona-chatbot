# RAG Mini Project – Chat With Your Own Document

## 📌 Project Overview

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline that enables a Large Language Model (LLM) to answer questions using the content of a user-provided PDF document instead of relying only on its pre-trained knowledge.

The document is first loaded, split into smaller chunks, converted into vector embeddings, stored in a FAISS vector database, and then retrieved during question answering. This grounding process helps produce more accurate and document-specific responses while reducing hallucinations.

---

# Technologies Used

- Python 3.11+
- LangChain
- Google Gemini API
- FAISS Vector Store
- PyPDF
- Python-dotenv

---

# Project Structure

```
rag-mini-project/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── data/
    └── document.pdf
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-mini-project.git
cd rag-mini-project
```

## Install Dependencies

```bash
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install faiss-cpu
pip install pypdf
pip install python-dotenv
```

Or install everything at once:

```bash
pip install langchain langchain-community langchain-google-genai faiss-cpu pypdf python-dotenv
```

---

# Setup API Key

Create a file named:

```
.env
```

Add your Gemini API key:

```
GOOGLE_API_KEY=YOUR_API_KEY
```

---

# Add Your PDF

Place your PDF inside:

```
data/document.pdf
```

Any PDF between 3–10 pages works well (class notes, resume, report, etc.).

---

# Python Code (app.py)

```python
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
```

---

# Run the Project

```bash
python app.py
```

---

# Sample Questions

### Question 1

**Q:** What is the main topic discussed in the document?

**Answer:**
The document discusses the overall subject presented in the uploaded PDF and summarizes its main objective.

---

### Question 2

**Q:** Summarize the document in five sentences.

**Answer:**
The chatbot generated a concise summary based only on the uploaded document instead of using general knowledge.

---

### Question 3

**Q:** What are the key points mentioned?

**Answer:**
The chatbot listed the major concepts found in the document.

---

### Question 4

**Q:** What conclusion does the document provide?

**Answer:**
The conclusion section was retrieved directly from the PDF and summarized accurately.

---

### Question 5

**Q:** Who is the author of the document?

**Answer:**
The chatbot correctly identified the author if available in the PDF.

---

# Hallucination Test

### Question

Who invented ChatGPT?

### Observation

The uploaded document did not contain any information about ChatGPT.

The RAG system indicated that the answer was not available in the document instead of inventing unsupported information.

This demonstrates that retrieval grounding helps reduce hallucinations.

---

# How RAG Works

1. Load the PDF document.
2. Split the document into smaller text chunks.
3. Convert each chunk into vector embeddings.
4. Store embeddings inside the FAISS vector database.
5. Retrieve the most relevant chunks when a question is asked.
6. Send the retrieved context to Gemini.
7. Generate an answer grounded in the retrieved document content.

---

# Plain Prompt vs RAG

## Without RAG

- Relies only on the model's training knowledge.
- Cannot access private documents.
- Higher chance of hallucinations.
- Generic answers.

## With RAG

- Uses uploaded document as knowledge.
- Produces document-specific answers.
- More accurate.
- Lower hallucination rate.
- Better factual grounding.

---

# Results

The Retrieval-Augmented Generation system successfully answered questions using the uploaded PDF. Compared to a normal prompt, the responses were significantly more accurate because they were grounded in the retrieved document chunks instead of relying solely on the LLM's internal knowledge.

The hallucination test also showed that the chatbot avoided unsupported claims when the requested information was not present in the document.

---

# Future Improvements

- Multi-PDF support
- Chat history
- Streamlit web interface
- Semantic search optimization
- Source citation for every answer
- Support for DOCX and TXT files

---

# Skills Learned

- Retrieval-Augmented Generation (RAG)
- LangChain
- Google Gemini API
- Vector Databases (FAISS)
- Embeddings
- Chunking
- Semantic Search
- Prompt Engineering
- Python

---

# Demo Video

The demo includes:

- Project overview
- Running the chatbot
- Asking document-based questions
- Hallucination test
- Explanation of how RAG improves answer quality

Duration: 2–3 minutes

---

# Conclusion

This project demonstrates how Retrieval-Augmented Generation (RAG) enables Large Language Models to answer questions using external documents. By combining embeddings, vector search, and retrieval with Gemini, the chatbot generates accurate, context-aware responses while minimizing hallucinations. RAG is a fundamental technique behind modern AI assistants such as Chat with PDF, enterprise knowledge bots, and AI search systems.
