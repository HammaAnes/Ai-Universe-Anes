from langchain_community.document_loaders import TextLoader
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
import os


# Load all chunks
chunks = []
folder_path = "chunks"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(folder_path, filename), encoding='utf-8')
        chunks.extend(loader.load())

# Embed and store in FAISS
embeddings = CohereEmbeddings(cohere_api_key="Q5FTyYuGJiCOZLh7lFXoEsf8ytRuRyHgDMZU1xX7", model="embed-english-v3.0")  # or HuggingFaceEmbeddings() for local models
db = FAISS.from_documents(chunks, embeddings)
db.save_local("microsoft_vectorstore")

