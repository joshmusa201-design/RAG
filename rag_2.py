# v2 of the rag. This is going to be a finance new rag. Wanted using a api to embedd my text because v1 was slow and its because am running it locally, but couldnt stand the api cost so am still gonna be running it locally here
import httpx
from pathlib import Path
import streamlit as stl
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb 
from huggingface_hub import InferenceClient
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient

load_dotenv()

# uing doclings library to create a structured document for the RAG model
import pymupdf4llm


safe = input("Ask me a question🤗: ")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_KEY"])
response = tavily_client.search(safe)

result = response["results"][0]["content"]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True)
bucket = []


texts = text_splitter.split_text(result)
for i, text in enumerate(texts):
    moving = f"{[text]}\n"
    bucket.append(moving)

ids = [f"id{y + 1}" for y in range(len(bucket))]
meta = [{"source":f"source{u + 1}"} for u in range(len(bucket))]
 

print("CHUNKING:", bucket)

# using a pre-trained model from HuggingFace to generate embeddings for the sentences
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = bucket
embeddings = model.encode(sentences)

#using ChromaDB to store the embeddings and sentences in a collection for retrieval
chroma_client = chromadb.Client()

collection = chroma_client.create_collection("my_collection")


# adding the embeddings and sentences to the collection(chromaDB)
collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=sentences,
    metadatas= meta#[{"source": "source1"}, {"source": "source2"}, {"source": "source3"}, {"source": "source4"}, {"source": "source5"}, {"source": "source6"}, {"source": "source7"}, {"source": "source7"}]
    )

class RAGModel:
    def __init__(self, collection, user):
        self.collection = collection
        self.user = user

    # retrieving the most relevant context from the collection based on the user's question
    def retrieve(self):
        query_embedding = model.encode(self.user).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        cool = results["documents"][0]
        answer = " ".join(cool)
        return answer

    # generating an answer to the user's question based on the retrieved context {answer}, from a pre-trained model in HuggingFace using the Inference API
    def generate_answer(self):
        try:
            client = InferenceClient(
                api_key=os.environ["HF_TOKEN"],
            )

            completion = client.chat.completions.create(
                model="Qwen/Qwen3-32B:nscale",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },

                    {
                        "role": "user",
                        "content": f"""Answer the user using the retrieved context." 
                            "Do not mention the context, retrieval, or how you found the answer."
                            "If the answer isn't in the context, say you don't know."
                            "User: {self.user}"
                            Context: {self.retrieve()}"""
                    },
                ],
            )
            return completion.choices[0].message.content
        except (httpx.ConnectTimeout, httpx.ConnectError) as e:
            return "Sorry, I'm having trouble generating an answer right now, check your internet connection and try again later."



joshua = RAGModel(collection, safe)
running = joshua.generate_answer()
print("Generated answer:", running)