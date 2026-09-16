# 🧠 RAG News Question Answering System

A Python-based Retrieval-Augmented Generation (RAG) project that retrieves relevant information before using a Large Language Model (LLM) to generate answers.

This project contains two versions, showing my progression from a document-based RAG system to a news-focused RAG system using web search.

---

## 🚀 RAG V1 — Document RAG

RAG V1 uses an Apple document as the knowledge source.

### Pipeline

```text
Apple Document
      ↓
PyMuPDF4LLM
      ↓
Text Extraction
      ↓
LangChain Chunking
      ↓
Hugging Face Embeddings
      ↓
ChromaDB
      ↓
Similarity Search
      ↓
Hugging Face LLM
      ↓
Answer

## 📰 RAG V2 — News RAG

RAG V2 extends the project from a static document to web-based information retrieval.

Instead of relying on one document, the system uses the Tavily API to search the web for relevant news and information based on the user's question.

### Pipeline
User Question
      ↓
Tavily Web Search
      ↓
Relevant News / Web Content
      ↓
Text Processing
      ↓
Chunking
      ↓
Hugging Face Embeddings
      ↓
ChromaDB
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
Hugging Face LLM
      ↓
Answer