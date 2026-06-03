from embeddings.embeddings import build_embeddings
from embeddings.similarity_check import search
from chunking.chunking import smart_chunk
from chunking.doc_loader import load_markdown_files

docs = load_markdown_files("../corpus/docker/manuals/subscription")

all_chunks = []

print(f"Loaded {len(docs)} documents.")

for doc in docs:
    chunked = smart_chunk(doc)
    print(f"Document: {doc['path']} - Chunks: {len(chunked)}")
    print(chunked)  # Print the first chunk for inspection
    all_chunks.extend(chunked)

vector_db = build_embeddings(all_chunks)

results = search("Can I transfer subscription?", vector_db)

for r in results:
    print("\nSCORE:", r["score"])
    print(r["metadata"])
    print(r["text"][:300])


test_queries = [
    "How to accommodate team changes",
    "How to scale consumption in docker?",
    "How to upgrade my subscription?",
]


def evaluate(vector_db):
    for q in test_queries:
        results = search(q, vector_db)

        print("\nQUERY:", q)
        print("TOP RESULT SCORE:", results[0]["score"])
        print("TOP SOURCE:", results[0]["metadata"])


evaluate(vector_db)
