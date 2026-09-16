# 🧠 RAG News Question Answering System

A Python-based Retrieval-Augmented Generation (RAG) project that retrieves relevant information from documents and the web before using a Large Language Model (LLM) to generate answers.

This project contains two versions, showing my progression from a document-based RAG system to a news-focused RAG system using web search.

## 🚀 RAG V1 — Document RAG

RAG V1 uses an Apple document as the knowledge source.

### Pipeline

Apple Document → PyMuPDF4LLM → Text Extraction → LangChain Chunking → Hugging Face Embeddings → ChromaDB → Similarity Search → Hugging Face LLM → Answer

### Technologies

- Python
- PyMuPDF4LLM
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Hugging Face LLM

The system extracts text from the document, splits it into smaller chunks, converts the chunks into embeddings, stores them in ChromaDB, retrieves the most relevant chunks based on the user's question, and passes the retrieved context to the LLM to generate an answer.

## 📰 RAG V2 — News RAG

RAG V2 extends the project from a static document to web-based information retrieval.

Instead of relying on one document, the system uses the Tavily API to search the web for relevant news and information based on the user's question.

### Pipeline

User Question → Tavily Web Search → Relevant News / Web Content → Text Processing → Chunking → Hugging Face Embeddings → ChromaDB → Similarity Search → Relevant Context → Hugging Face LLM → Answer

### Technologies

- Python
- Tavily API
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Hugging Face LLM

The goal of RAG V2 is to make the system more useful for questions about current events and news by retrieving relevant information from the web before generating an answer.

## 🧠 What I Learned

Building both versions helped me understand:

- Document extraction
- Text chunking
- Embeddings
- Vector databases
- Similarity search
- Information retrieval
- Web search
- LLM context retrieval
- Retrieval-Augmented Generation
- Debugging RAG pipelines

One of the biggest lessons I learned was that RAG quality depends on the entire pipeline. If important information is lost during document extraction or retrieval, the LLM cannot reliably recover it later.

## 🔮 Future Improvements

- Improve retrieval quality
- Add source citations
- Add metadata filtering
- Experiment with reranking
- Add RAG evaluation
- Support more news sources
- Improve handling of recent information

## 👤 Author

Built by Joshua while learning AI/ML and Retrieval-Augmented Generation.
