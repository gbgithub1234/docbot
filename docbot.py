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
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# --- INITIALIZATION ---
openai.api_key = OPENAI_API_KEY
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

EMBED_MODEL = "text-embedding-ada-002"

# --- HELPER FUNCTIONS ---

def load_pdf(file: BytesIO):
    reader = PdfReader(file)
    return [page.extract_text() for page in reader.pages]

def load_docx(file: BytesIO):
    doc = Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text(texts, chunk_size=1000, chunk_overlap=100):
    chunks = []
    for text in texts:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
    return chunks

def embed_texts(texts):
    response = openai.embeddings.create(
        input=texts,
        model=EMBED_MODEL
    )
    embeddings = [d.embedding for d in response.data]
    return embeddings

def store_embeddings(texts, embeddings, source_name, batch_size=50):
    ids = [str(uuid4()) for _ in embeddings]
    metadata = [
        {"source": source_name, "text": text[:1000]}  # Save first 1000 characters in metadata
        for text in texts
    ]
    vectors = [
        {
            "id": id_,
            "values": embedding,
            "metadata": meta
        }
        for id_, embedding, meta in zip(ids, embeddings, metadata)
    ]

    # Batch upserts to avoid payload errors
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

def retrieve_contexts(query, top_k=5):
    query_embed = openai.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    ).data[0].embedding

    results = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    contexts = [match.metadata.get('text', '') for match in results.matches if 'text' in match.metadata]
    return contexts

def generate_answer(contexts, query):
    context_text = "\n---\n".join(contexts)
    prompt = f"Use the following context to answer the question.\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

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
        contexts = retrieve_contexts(query)
        answer = generate_answer(contexts, query)

        st.write("### Answer:")
        st.write(answer)

        with st.expander("See retrieved document sections"):
            for i, context in enumerate(contexts):
                st.write(f"**Section {i+1}:**\n{context}")
