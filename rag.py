# am gonna be creating a RAG model for question answering using langchain and huggingface transformers. The model will be trained on a dataset of questions and answers, and will be able to generate answers to new questions based on the training data.
import httpx
import streamlit as stl
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb 
from huggingface_hub import InferenceClient
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# uing doclings library to create a structured document for the RAG model
source = r"C:\Users\LENOVO\Desktop\rag\2026_apple_reports.pdf"  # file path or URL
doc = DocumentConverter().convert(source=source).document
me = doc.export_to_markdown()  # output: "### Docling Technical Report[...]"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0, add_start_index=True)
bucket = []

texts = text_splitter.split_text(me)
for i, text in enumerate(texts):
    moving = f"Chunk {i + 1}: {[text]}\n"
    bucket.append(moving)

print(bucket)


load_dotenv()

user = input("Ask me a question🤗: ")

# using a pre-trained model from HuggingFace to generate embeddings for the sentences
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = bucket
embeddings = model.encode(sentences)
print(sentences)

#using ChromaDB to store the embeddings and sentences in a collection for retrieval
chroma_client = chromadb.Client()

collection = chroma_client.create_collection("my_collection")


# adding the embeddings and sentences to the collection(chromaDB)
collection.add(
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
    embeddings=embeddings.tolist(),
    documents=sentences,
    metadatas=[{"source": "source1"}, {"source": "source2"}, {"source": "source3"}, {"source": "source4"}, {"source": "source5"}, {"source": "source6"}, {"source": "source7"}, {"source": "source8"}]
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



