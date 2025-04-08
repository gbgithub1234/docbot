# docbot.py

import streamlit as st
import openai
import pinecone
import pandas as pd
import os
from typing import List
from uuid import uuid4
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "docbot-index"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-ada-002"

# --- INITIALIZATION ---
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create or connect to Pinecone index
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536)
index = pinecone.Index(PINECONE_INDEX_NAME)

# --- HELPER FUNCTIONS ---

def load_pdf(file: BytesIO) -> List[str]:
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return text

def load_docx(file: BytesIO) -> List[str]:
    doc = Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text(texts: List[str], chunk_size: int = 500) -> List[str]:
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        input=texts,
        model=EMBED_MODEL
    )
    embeddings = [d['embedding'] for d in response['data']]
    return embeddings

def store_embeddings(texts: List[str], embeddings: List[List[float]], source_name: str):
    ids = [str(uuid4()) for _ in embeddings]
    metadata = [{"source": source_name, "text": text} for text in texts]
    to_upsert = list(zip(ids, embeddings, metadata))
    index.upsert(vectors=to_upsert)

# --- STREAMLIT APP ---
st.set_page_config(page_title="DocBot", layout="wide")
st.title("📄 DocBot - Smart Document Search")

# Upload document
uploaded_file = st.file_uploader("Upload a PDF or Word Document", type=["pdf", "docx"])

if uploaded_file:
    with st.spinner("Processing document..."):
        if uploaded_file.name.endswith(".pdf"):
            texts = load_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            texts = load_docx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        chunks = split_text(texts)
        embeddings = embed_texts(chunks)
        store_embeddings(chunks, embeddings, uploaded_file.name)

    st.success(f"Uploaded and indexed: {uploaded_file.name}")

# Ask a question
query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Searching for answers..."):
        # Embed query
        query_embed = openai.embeddings.create(input=[query], model=EMBED_MODEL)['data'][0]['embedding']

        # Query Pinecone
        results = index.query(vector=query_embed, top_k=5, include_metadata=True)

        # Build context
        contexts = [match['metadata']['text'] for match in results['matches'] if 'text' in match['metadata']]
        context_text = "\n---\n".join(contexts)

        # Ask LLM
        prompt = f"Use the following context to answer the question.\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content

        st.write("### Answer:")
        st.write(answer)

        with st.expander("See retrieved document sections"):
            for i, context in enumerate(contexts):
                st.write(f"**Section {i+1}:**\n{context}")
