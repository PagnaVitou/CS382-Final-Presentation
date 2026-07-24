"""
RAG-Based AI Search System — enhanced interface.

Run with:
    streamlit run app.py

This gives you a working, end-to-end RAG demo: document loading, semantic
retrieval with real embeddings, and LLM-powered answers grounded in your documents.
"""

import time
import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/cs_lecture_notes_all_100"

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store(chunk_size: int = 100, overlap: int = 20):
    """Load documents and build the vector store."""
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs, chunk_size=chunk_size, overlap=overlap)
    store = VectorStore(use_embeddings=True)
    store.build(chunks)
    return store, docs, chunks


st.title("🔎 RAG-Based AI Search System")
st.caption("Ask questions about the indexed CS lecture notes. Answers are grounded in the source material.")

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Retrieval")
    top_k = st.slider("Number of chunks to retrieve (top-k)", min_value=1, max_value=15, value=3,
                      help="How many relevant passages to use when generating answers")
    
    chunk_size = st.slider("Chunk size (words)", min_value=50, max_value=200, value=100, step=10,
                          help="Size of text chunks for retrieval. Smaller = more granular, Larger = more context")
    
    overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=50, value=20, step=5,
                       help="Overlap between consecutive chunks to preserve context")
    
    st.divider()
    
    st.subheader("Generation")
    mode = st.radio("Answer mode", ["extractive", "llm"], 
                   help="Extractive: no API needed, just returns top chunks. LLM: AI-generated answers (requires ANTHROPIC_API_KEY)")
    
    st.divider()
    
    st.subheader("📊 Index Info")
    store, docs, chunks = load_store(chunk_size=chunk_size, overlap=overlap)
    st.info(f"📄 **{len(docs)}** documents → **{len(chunks)}** chunks")
    
    with st.expander("View indexed documents"):
        for d in docs:
            st.caption(f"• {d['title']}")

query = st.text_input(
    "Your question",
    placeholder="e.g., What is a hash table? How does content-based filtering work?",
    key="query_text",
)

# Use lookups from filename and title → document so source downloads can serve the exact file.
docs_by_filename = {}
docs_by_title = {}
for d in docs:
    if isinstance(d, dict):
        if "filename" in d:
            docs_by_filename[d["filename"]] = d
        docs_by_title[d.get("title", "")] = d

col1, col2, col3 = st.columns(3)
with col1:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    if st.button("📋 Example Queries", use_container_width=True):
        st.session_state.show_examples = not st.session_state.get("show_examples", False)

if query.strip() and search_clicked:
    start_time = time.time()
    
    retrieved = store.query(query, top_k=top_k)
    answer = generate_answer(query, retrieved, mode=mode)
    elapsed = time.time() - start_time
    
    st.subheader("✅ Answer")
    st.write(answer)
    
    st.subheader("📚 Retrieved Sources")
    if not retrieved:
        st.warning("No relevant sources found.")
    else:
        for i, (chunk, score) in enumerate(retrieved, 1):
            with st.expander(f"**Source {i}**: {chunk.doc_title} · Similarity: {score:.3f}"):
                st.write(chunk.text)
                st.caption(f"Chunk ID: {chunk.chunk_id}")
                # Use getattr with fallback for compatibility with cached objects
                doc_filename = getattr(chunk, 'doc_filename', None)
                source_doc = docs_by_filename.get(doc_filename) if doc_filename else None
                if source_doc is None:
                    source_doc = docs_by_title.get(chunk.doc_title)
                if source_doc is not None:
                    file_name = source_doc.get("filename", f"{chunk.doc_title}.txt")
                    st.download_button(
                        label="Download full source document",
                        data=source_doc["text"],
                        file_name=file_name,
                        mime="text/plain",
                        key=f"download_{i}_{chunk.chunk_id}",
                    )
                else:
                    st.info("Full document not available for download.")
    
    st.divider()
    st.caption(f"⏱️ Search completed in {elapsed:.2f}s | Mode: {mode}")
    
elif search_clicked:
    st.warning("⚠️ Please type a question first.")

if st.session_state.get("show_examples", False):
    st.divider()
    st.subheader("💡 Example Queries")
    examples = [
        "What is dynamic programming?",
        "How does binary search work?",
        "Explain sorting algorithms.",
        "What are hash tables used for?",
        "What is object-oriented programming?",
        "How does recursion work?",
        "What is the difference between arrays and linked lists?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.query_text = ex
