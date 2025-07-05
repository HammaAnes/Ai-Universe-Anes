import streamlit as st
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere.chat_models import ChatCohere
from langchain.prompts import PromptTemplate

# --- Initialize Embeddings and Vector Store ---
embeddings = CohereEmbeddings(
    cohere_api_key="Q5FTyYuGJiCOZLh7lFXoEsf8ytRuRyHgDMZU1xX7",
    model="embed-english-v3.0"
)

db = FAISS.load_local("brainywriter_vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# --- Initialize LLM and Prompt ---
llm = ChatCohere(
    temperature=0,
    cohere_api_key="Q5FTyYuGJiCOZLh7lFXoEsf8ytRuRyHgDMZU1xX7",
    model="command-a-03-2025"
)

template = """
Tu es un assistant IA expert de la plateforme BrainyWriter.

Tu dois :
- Répondre uniquement aux questions en lien avec BrainyWriter (fonctionnalités, fonctionnement, templates, tokens, collaboration, etc.).
- Être poli, clair et utile.
- Refuser poliment toute question hors sujet.

Utilise le contexte ci-dessous pour répondre.

---
Chat History:
{chat_history}

Context:
{context}
---
Question: {question}
Réponse :
"""



prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# --- Streamlit UI ---
st.set_page_config(page_title="BrainyWriter Chatbot", page_icon="🧠")

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("💬 BrainyWriter RAG Chatbot with Memory")
st.markdown("Ask me anything about BrainyWriter.")

user_input = st.text_input("🔍 Your question")

if user_input:
    with st.spinner("Thinking..."):
        # Call the conversational chain
        result = qa_chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })

        # Save this interaction in the session
        st.session_state.chat_history.append((user_input, result["answer"]))

        # Show the response
        st.success("✅ Answer:")
        st.write(result["answer"])

        # Optional: Show context chunks
        if "source_documents" in result:
            with st.expander("📄 View retrieved context"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Source**: `{doc.metadata.get('source', 'unknown')}`")
                    st.text(doc.page_content)

