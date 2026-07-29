"""
RAG-Based AI Search System — enhanced interface.

Run with:
    streamlit run app.py

This gives you a working, end-to-end RAG demo: document loading, semantic
retrieval with real embeddings, and LLM-powered answers grounded in your documents.
"""

import time
import streamlit as st
import re

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


st.title("Find the your NOTEES! 🔎")
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
    # Relevance threshold to avoid answering on low-confidence retrievals
    st.subheader("Relevance")
    relevance_threshold = st.slider("Minimum top similarity to answer (threshold)", min_value=0.00, max_value=0.50, value=0.15, step=0.01,
                                    help="If the top retrieved chunk's similarity is below this value the app will not generate an answer unless forced.")
    force_answer = st.checkbox("Force answer even if below threshold", value=False)
    stricter_relevance = st.checkbox("Use stricter relevance check (avg score)", value=True,
                                     help="Also require the average similarity of retrieved chunks to be reasonably high to avoid weak, scattered matches.")
    st.subheader("Domain Filtering")
    enforce_domain_filter = st.checkbox("Enforce domain filter (block out-of-topic queries)", value=True)
    domain_overlap_threshold = st.slider("Minimum query↔corpus token overlap", min_value=0.0, max_value=1.0, value=0.20, step=0.01,
                                         help="Fraction of distinct query tokens that must appear in the indexed corpus to be considered in-domain.")
    
    st.divider()
    
    st.subheader("📊 Index Info")
    store, docs, chunks = load_store(chunk_size=chunk_size, overlap=overlap)
    st.info(f"📄 **{len(docs)}** documents → **{len(chunks)}** chunks")
    
    with st.expander("View indexed documents"):
        for d in docs:
            st.caption(f"• {d['title']}")

selected_example = st.session_state.get("selected_example", "")
query = st.text_input("Enter your question", key="query_text")
query = query if isinstance(query, str) else ""
query = query.strip()

# Use lookups from filename and title → document so source downloads can serve the exact file.
docs_by_filename = {}
docs_by_title = {}
for d in docs:
    if isinstance(d, dict):
        if "filename" in d:
            docs_by_filename[d["filename"]] = d
        docs_by_title[d.get("title", "")] = d

# Build a simple corpus vocabulary for lightweight domain checking
stopwords = {
    "the", "is", "in", "and", "to", "of", "a", "an", "for", "on", "with", "by", "as", "that",
    "this", "are", "be", "or", "it", "from", "at", "which",
}
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

corpus_words = set()
for d in docs:
    text = d.get("text", "") if isinstance(d, dict) else ""
    for t in tokenize(text):
        if t not in stopwords:
            corpus_words.add(t)

col1, col2, col3 = st.columns(3)
with col1:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    if st.button("📋 Example Queries", use_container_width=True):
        st.session_state.show_examples = not st.session_state.get("show_examples", False)

if query.strip() and search_clicked:
    start_time = time.time()
    
    # Domain check: refuse out-of-domain queries if enabled
    query_tokens = [t for t in tokenize(query) if t not in stopwords]
    distinct_q = set(query_tokens)
    overlap = 0.0
    if distinct_q:
        overlap = sum(1 for t in distinct_q if t in corpus_words) / len(distinct_q)

    if enforce_domain_filter and (distinct_q and overlap < domain_overlap_threshold) and not force_answer:
        st.warning(f"Out-of-domain query detected (token overlap {overlap:.2f} < {domain_overlap_threshold:.2f}).")
        st.info("This query appears outside the indexed CS lecture notes. Try rephrasing or uncheck 'Enforce domain filter' to force an answer.")
        # Show short preview of nearest passages so user can inspect
        retrieved = store.query(query, top_k=top_k)
        if retrieved:
            preview = []
            for chunk, score in retrieved:
                preview.append(f"[{chunk.doc_title}, score={score:.3f}]\n{chunk.text}\n")
            st.code("\n---\n".join(preview))
        elapsed = time.time() - start_time
        answer = "[Query refused: detected as out-of-domain relative to the indexed corpus.]"
        st.subheader("✅ Answer")
        st.write(answer)
    else:
        retrieved = store.query(query, top_k=top_k)
        elapsed = time.time() - start_time

    # Check top similarity and apply relevance threshold
    st.subheader("✅ Answer")
    if not retrieved:
        st.warning("No relevant sources found.")
        answer = "No relevant passages were found for that query."
        st.write(answer)
    else:
        scores = [s for (_, s) in retrieved]
        top_score = scores[0]
        avg_score = sum(scores) / len(scores)
        # Decide refusal if top score is below threshold or average score is too low
        avg_threshold = relevance_threshold * 0.6
        low_confidence = (top_score < relevance_threshold) or (stricter_relevance and avg_score < avg_threshold)
        if low_confidence and not force_answer:
            st.warning(
                f"Low confidence: top retrieved similarity {top_score:.3f} (avg {avg_score:.3f}) is below the threshold {relevance_threshold:.3f}."
            )
            st.info("Options: rephrase your question, increase the threshold sensitivity, or check the retrieved sources below.")
            # Show an extractive preview so the user can inspect the retrieved passages
            preview = []
            for chunk, score in retrieved:
                preview.append(f"[{chunk.doc_title}, score={score:.3f}]\n{chunk.text}\n")
            st.code("\n---\n".join(preview))
            answer = "[Not answered due to low confidence — see retrieved passages above.]"
            st.write(answer)
        else:
            answer = generate_answer(query, retrieved, mode=mode)
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
            st.session_state.selected_example = ex
            st.session_state.show_examples = False
