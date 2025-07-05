from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_cohere.embeddings import CohereEmbeddings

# Load and split the FAQ into individual Q&A chunks
def load_brainywriter_faq(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        if line.startswith("Q:") and i + 1 < len(lines):
            question = line
            answer = lines[i + 1].strip()
            content = f"{question}\n{answer}"
            doc = Document(page_content=content, metadata={"source": "brainywriter_faq.txt"})
            chunks.append(doc)
            i += 3  # Skip question, answer, and empty separator
        else:
            i += 1

    return chunks

# Initialize embeddings
embeddings = CohereEmbeddings(
    cohere_api_key="Q5FTyYuGJiCOZLh7lFXoEsf8ytRuRyHgDMZU1xX7",  # 🔐 Replace with your real API key
    model="embed-english-v3.0"
)

# Load chunks from file
faq_chunks = load_brainywriter_faq("BrainyWriter_FAQ_Chatbot.txt")

# Create and save vectorstore
db = FAISS.from_documents(faq_chunks, embeddings)
db.save_local(r"brainy\brainywriter_vectorstore")

print(f"✅ Indexed {len(faq_chunks)} Q&A pairs into vectorstore.")
