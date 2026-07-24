"""
Evaluation module for the RAG system.

Tests retrieval quality and generation accuracy on a curated set of queries.
Provides both qualitative and quantitative metrics.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass

from .ingest import Chunk, load_documents, build_chunk_records
from .embed_store import VectorStore
from .generate import generate_answer


@dataclass
class TestQuery:
    """A test query with expected relevant topics."""
    query: str
    expected_topics: List[str]
    description: str


# Curated test queries
TEST_QUERIES = [
    TestQuery(
        query="What is a hash table and how does it work?",
        expected_topics=["Hash Tables", "Data Structures", "Hashing"],
        description="Tests basic data structure knowledge"
    ),
    TestQuery(
        query="Explain the difference between arrays and linked lists.",
        expected_topics=["Arrays", "Linked Lists", "Data Structures"],
        description="Tests comparative understanding of data structures"
    ),
    TestQuery(
        query="How does binary search differ from linear search?",
        expected_topics=["Searching Algorithms", "Binary Search", "Time Complexity"],
        description="Tests algorithm differentiation and performance concepts"
    ),
    TestQuery(
        query="What are sorting algorithms and their time complexities?",
        expected_topics=["Sorting Algorithms", "Time Complexity", "Big O Notation"],
        description="Tests algorithm analysis knowledge"
    ),
    TestQuery(
        query="Explain recursion and when to use it.",
        expected_topics=["Recursion", "Base Case", "Stack"],
        description="Tests recursive programming concepts"
    ),
    TestQuery(
        query="What is dynamic programming used for?",
        expected_topics=["Dynamic Programming", "Optimization", "Memoization"],
        description="Tests advanced algorithm technique understanding"
    ),
    TestQuery(
        query="Describe object-oriented programming principles.",
        expected_topics=["OOP", "Classes", "Inheritance", "Polymorphism", "Encapsulation"],
        description="Tests OOP fundamentals"
    ),
    TestQuery(
        query="What are the steps in the software development lifecycle?",
        expected_topics=["Software Engineering", "Requirements", "Design", "Testing"],
        description="Tests software engineering process knowledge"
    ),
]


def evaluate_retrieval(store: VectorStore, query: str, expected_topics: List[str], top_k: int = 3) -> Dict:
    """
    Evaluate retrieval quality for a query.
    
    Args:
        store: The vector store to query
        query: The test query
        expected_topics: List of topics that should appear in results
        top_k: Number of results to retrieve
    
    Returns:
        Dict with retrieval metrics
    """
    retrieved = store.query(query, top_k=top_k)
    
    scores = [score for _, score in retrieved]
    titles = [chunk.doc_title for chunk, _ in retrieved]
    texts = [chunk.text for chunk, _ in retrieved]
    
    relevant_count = sum(
        1 for title in titles 
        if any(topic.lower() in title.lower() for topic in expected_topics)
    )
    
    return {
        "query": query,
        "num_retrieved": len(retrieved),
        "avg_similarity": sum(scores) / len(scores) if scores else 0,
        "min_similarity": min(scores) if scores else 0,
        "max_similarity": max(scores) if scores else 0,
        "relevant_count": relevant_count,
        "relevant_docs": titles,
        "snippets": texts[:2],  # Show first 2 snippets
    }


def evaluate_generation(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    """Generate an answer and return it for evaluation."""
    return generate_answer(query, retrieved, mode=mode)


def run_evaluation(data_folder: str = "data/cs_lecture_notes_all_100", chunk_size: int = 100, overlap: int = 20) -> None:
    """Run full evaluation suite and print results."""
    print("\n" + "="*80)
    print("RAG SYSTEM EVALUATION")
    print("="*80 + "\n")
    
    # Load data
    print("Loading documents and building vector store...")
    docs = load_documents(data_folder)
    chunks = build_chunk_records(docs, chunk_size=chunk_size, overlap=overlap)
    store = VectorStore(use_embeddings=True)
    store.build(chunks)
    print(f"[OK] Loaded {len(docs)} documents, created {len(chunks)} chunks\n")
    
    # Retrieval evaluation
    print("="*80)
    print("RETRIEVAL EVALUATION")
    print("="*80 + "\n")
    
    retrieval_scores = []
    for i, test in enumerate(TEST_QUERIES, 1):
        result = evaluate_retrieval(store, test.query, test.expected_topics, top_k=3)
        retrieval_scores.append(result)
        
        print(f"Query {i}: {test.query}")
        print(f"Description: {test.description}")
        print(f"Expected topics: {', '.join(test.expected_topics)}")
        print(f"Avg similarity score: {result['avg_similarity']:.3f}")
        print(f"Relevant docs found: {result['relevant_count']}/3")
        print(f"Retrieved: {', '.join(result['relevant_docs'])}")
        print()
    
    # Calculate aggregate retrieval metrics
    avg_similarity = sum(r['avg_similarity'] for r in retrieval_scores) / len(retrieval_scores)
    relevance_rate = sum(r['relevant_count'] for r in retrieval_scores) / len(retrieval_scores)
    
    print("-"*80)
    print(f"Average similarity across all queries: {avg_similarity:.3f}")
    print(f"Average relevance rate (docs matching expected topics): {relevance_rate:.2f}/3")
    print()
    
    # Generation evaluation (brief)
    print("="*80)
    print("GENERATION EVALUATION (Extractive Mode)")
    print("="*80 + "\n")
    
    for i, test in enumerate(TEST_QUERIES[:3], 1):  # Show first 3 only
        retrieved = store.query(test.query, top_k=3)
        answer = evaluate_generation(test.query, retrieved, mode="extractive")
        
        print(f"Query {i}: {test.query}\n")
        print("Answer (first 300 chars):")
        print(answer[:300] + ("..." if len(answer) > 300 else "") + "\n")
        print("-"*40 + "\n")
    
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_evaluation()
