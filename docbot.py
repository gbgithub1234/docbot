# --- Imports ---
import streamlit as st
import openai
import pinecone
from io import BytesIO

# --- Load Secrets ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# --- Initialize Clients ---
openai.api_key = OPENAI_API_KEY
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

EMBED_MODEL = "text-embedding-ada-002"

# --- Helper Functions ---
def retrieve_contexts(query, top_k=10):
    query_embed = openai.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    ).data[0].embedding

    results = index.query(vector=query_embed, top_k=top_k, include_metadata=True)
    contexts = [match.metadata.get('text', '') for match in results.matches if 'text' in match.metadata]
    sources = [match.metadata.get('source', 'Unknown') for match in results.matches if 'source' in match.metadata]
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

# --- Streamlit App Setup ---
st.set_page_config(page_title="DocBot", layout="wide")

# Top intro
with st.expander("Show/hide details"):
    st.write("""
    - Created by Glen Brauer, Business Analyst in AAE (glenb@sfu.ca)
    - PROBLEM: Document-based information is located in many places, taking time to find.
    - SOLUTION: Provide a one-stop search for document-based information.
    - Powered by AI and [Pinecone Vector Storage](https://www.pinecone.io/)
    - Access [sample documents](https://drive.google.com/drive/u/0/folders/1gTD-OiqH5Bg3-ZqVuur9q8h-AGIzOlB7)
    """)

st.header("SFU Document Chatbot 2.0 (beta)")

# --- Question Box ---
st.markdown("**Press ENTER or click the Search button below** 👇")

# Input box
query = st.text_input("Ask a question about your documents:", key="user_query")

# Button
search_button = st.button("🔍 Search")

# If user entered a query AND either pressed Enter or clicked the button
if (query and search_button) or (query and not search_button and st.session_state.get("search_triggered") is None):
    st.session_state["search_triggered"] = True  # Mark that we already triggered once

    with st.spinner("Searching for answers..."):
        contexts, sources = retrieve_contexts(query)

        if contexts:
            answer = generate_answer(contexts, query)

            st.write("### Answer:")
            st.write(answer)

            if sources:
                unique_sources = sorted(set(sources))
                st.markdown("### 📄 Sources used:")
                for src in unique_sources:
                    st.markdown(f"- {src}")
        else:
            st.warning("⚠️ No relevant documents found. Please upload documents first (admin).")

# Reset trigger if button pressed manually
if search_button:
    st.session_state["search_triggered"] = None



# --- Sidebar: Uploaded Files ---
st.sidebar.title("📄 Uploaded Files")

uploaded_files = get_uploaded_files()

if isinstance(uploaded_files, str):
    st.sidebar.error(uploaded_files)
elif uploaded_files:
    for file in uploaded_files:
        st.sidebar.markdown(f"- {file}")
else:
    st.sidebar.info("No files found.")

