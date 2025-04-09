# docbot.py

import streamlit as st
import openai
import pinecone
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from uuid import uuid4

# --- LOAD SECRETS ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# --- INITIALIZATION ---
openai.api_key = OPENAI_API_KEY
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

EMBED_MODEL = "text-embedding-ada-002"

# --- HELPER FUNCTIONS ---

def retrieve_contexts(query, top_k=10):
    query_embed = openai.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    ).data[0].embedding

    results = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    contexts = []
    sources = []
    for match in results.matches:
        if 'text' in match.metadata:
            contexts.append(match.metadata['text'])
            if 'source' in match.metadata:
                sources.append(match.metadata['source'])
    return contexts, sources

def generate_answer(contexts, query):
    context_text = "\n---\n".join(contexts)
    prompt = f"Use the following context to answer the question.\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def get_uploaded_files():
    try:
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            return []
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=5000,
            include_metadata=True,
            include_values=False
        )
        files = set()
        for match in results.matches:
            if 'source' in match.metadata:
                files.add(match.metadata['source'])
        return sorted(list(files))
    except Exception as e:
        return f"Error retrieving uploaded files: {e}"

# --- STREAMLIT APP ---

st.set_page_config(page_title="DocBot", layout="wide")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Top introduction
with st.expander("Show/hide details"):
    st.write("""
    - created by Glen Brauer, Business Analyst in AAE (glenb@sfu.ca)
    - PROBLEM: document-based information is located in many places taking time to find
    - SOLUTION: provide a one-stop shopping resource for all document-based information
    - Leverages AI and [Pinecone vector storage](https://www.pinecone.io/)
    - Access [sample documents](https://drive.google.com/drive/u/0/folders/1gTD-OiqH5Bg3-ZqVuur9q8h-AGIzOlB7)
    """)

st.header("SFU Document Chatbot 2.0 (beta)")

# --- Chat History Section ---
st.markdown("### 🗨️ Chat History")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        if chat['sources']:
            st.markdown("**Sources:**")
            for src in chat['sources']:
                st.markdown(f"- {src}")
        st.markdown("---")
else:
    st.info("No conversation history yet. Start by asking a question!")

st.markdown("---")

# --- Question box ---
st.text_input("Ask a question about your uploaded documents:", key="user_query")

if st.session_state.get("user_query"):  # Only proceed if there's a non-empty input
    with st.spinner("Searching for answers..."):
        query = st.session_state["user_query"]
        contexts, sources = retrieve_contexts(query)

        if contexts:
            answer = generate_answer(contexts, query)

            # Save the chat into history
            st.session_state.chat_history.append({
                "user": query,
                "bot": answer,
                "sources": sorted(set(sources))
            })

            # 🛑 Clear the input box
            st.session_state["user_query"] = ""

            st.rerun()

        else:
            st.warning("⚠️ No relevant documents found. Please upload documents first (admin).")
            # Even if no docs found, clear input
            st.session_state["user_query"] = ""



# --- Sidebar: Uploaded Files ---
uploaded_files = get_uploaded_files()
file_count = len(uploaded_files) if isinstance(uploaded_files, list) else 0

with st.sidebar.expander(f"📄 Uploaded Files ({file_count})", expanded=True):
    st.subheader("Uploaded Files")
    if isinstance(uploaded_files, str):
        st.error(uploaded_files)
    elif uploaded_files:
        for file in uploaded_files:
            st.markdown(f"- {file}")
    else:
        st.info("No files found.")
