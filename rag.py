# am gonna be creating a RAG model for question answering using langchain and huggingface transformers. The model will be trained on a dataset of questions and answers, and will be able to generate answers to new questions based on the training data.

import httpx
import streamlit as stl
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import InferenceClient

load_dotenv()

user = input("Ask me a question🤗: ")

# using a pre-trained model from HuggingFace to generate embeddings for the sentences
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)


#using ChromaDB to store the embeddings and sentences in a collection for retrieval
chroma_client = chromadb.Client()

collection = chroma_client.create_collection("my_collection")


# adding the embeddings and sentences to the collection(chromaDB)
collection.add(
    ids=["id1", "id2", "id3", "id4"],
    embeddings=embeddings.tolist(),
    documents=sentences,
    metadatas=None
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
            return "Sorry, I'm having trouble generating an answer right now."



joshua = RAGModel(collection, user)
running = joshua.generate_answer()
print("Generated answer:", running)



