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

def load_pdf(file: BytesIO):
    reader = PdfReader(file)
    return [page.extract_text() for page in reader.pages]

def load_docx(file: BytesIO):
    doc = Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text(texts, chunk_size=1000, chunk_overlap=100):
    chunks = []
    for text in texts:
        text = text.strip()
        if len(text) <= chunk_size:
            chunks.append(text)
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - chunk_overlap
    return chunks

def embed_texts(texts):
    clean_texts = [text for text in texts if text.strip()]
    if not clean_texts:
        return [], []
    response = openai.embeddings.create(
        input=clean_texts,
        model=EMBED_MODEL
    )
    embeddings = [d.embedding for d in response.data]
    return clean_texts, embeddings

def store_embeddings(texts, embeddings, source_name, batch_size=50):
    vectors = []
    for text, embedding in zip(texts, embeddings):
        if text.strip() and embedding:
            vectors.append({
                "id": str(uuid4()),
                "values": embedding,
                "metadata": {"source": source_name, "text": text[:1000]}
            })
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

def retrieve_contexts(query, top_k=10):
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

# Initialize session state flags
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False
if "delete_triggered" not in st.session_state:
    st.session_state.delete_triggered = False
if "query" not in st.session_state:
    st.session_state.query = ""

# --- File upload ---
uploaded_file = st.file_uploader("Upload a PDF or Word Document", type=["pdf", "docx"])

if uploaded_file and not st.session_state.upload_complete:
    with st.spinner(f"Uploading and processing '{uploaded_file.name}'... Please wait."):
        try:
            texts = load_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else load_docx(uploaded_file)
            chunks = split_text(texts)
            clean_texts, embeddings = embed_texts(chunks)
            if clean_texts and embeddings:
                store_embeddings(clean_texts, embeddings, uploaded_file.name)
                st.success(f"✅ '{uploaded_file.name}' uploaded and indexed!")
                st.session_state.upload_complete = True
            else:
                st.error("⚠️ No valid text extracted from the uploaded document.")
        except Exception as e:
            st.error(f"Error during upload: {e}")

# --- Question box ---
query = st.text_input("Ask a question about your documents:", value=st.session_state.query, key="query")

if query:
    with st.spinner("Searching for answers..."):
        contexts = retrieve_contexts(query)
        if not contexts:
            st.warning("⚠️ No relevant documents found. Please upload documents first.")
        else:
            answer = generate_answer(contexts, query)
            st.write("### Answer:")
            st.write(answer)

            with st.expander("See retrieved document sections"):
                for i, context in enumerate(contexts):
                    st.write(f"**Section {i+1}:**\n{context}")

st.markdown("---")

# --- Sidebar: Uploaded Files + Delete ---

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

    st.markdown("---")
    st.subheader("🗑️ Delete Uploaded File")

    if isinstance(uploaded_files, list) and uploaded_files:
        selected_file = st.selectbox("Select a file to delete:", uploaded_files, key="delete_file")

        if st.button(f"Confirm Delete '{selected_file}'", key="confirm_delete"):
            with st.spinner(f"Deleting all vectors from '{selected_file}'..."):
                try:
                    index.delete(filter={"source": {"$eq": selected_file}})
                    st.success(f"✅ Deleted all vectors for '{selected_file}'.")
                    st.session_state.delete_triggered = True
                except Exception as e:
                    st.error(f"Error deleting vectors: {e}")
    else:
        st.info("No files available to delete.")

# --- Handle Delete and Refresh ---
if st.session_state.delete_triggered:
    st.session_state.query = ""
    st.session_state.delete_triggered = False
    st.rerun()
