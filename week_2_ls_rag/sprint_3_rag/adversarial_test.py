from embeddings.embeddings import build_embeddings
from embeddings.similarity_check import search
from chunking.chunking import smart_chunk
from chunking.doc_loader import load_markdown_files

docs = load_markdown_files("../corpus/docker/manuals/subscription")
docs += load_markdown_files("../corpus/fastapi/docs/deployment")
docs += load_markdown_files(
    "../corpus/streamlit/content/deploy/community-cloud/deploy-your-app"
)


all_chunks = []

print(f"Loaded {len(docs)} documents.")

for doc in docs:
    chunked = smart_chunk(doc)
    # print(f"Document: {doc['path']} - Chunks: {len(chunked)}")
    # print(chunked)  # Print the first chunk for inspection
    all_chunks.extend(chunked)

vector_db = build_embeddings(all_chunks)

test_queries = [
    "enterprise plan for large organizations",
    "What runtime should host FastAPI in production?",
    "How do I manage billing and add new users to a Docker team?",
    "How do I deploy FastAPI?",
    "How do I deploy FastAPI to AWS Lambda?"
    "Where should API keys be stored in Streamlit Cloud?",
]


def evaluate(vector_db):
    for q in test_queries:
        results = search(q, vector_db)

        print("\nQUERY:", q)
        print("TOP RESULT SCORE:", results[0]["score"])
        print("TOP SOURCE:", results[0]["metadata"])
        print("TOP TEXT:", results[0]["text"])
        print("-" * 50)


evaluate(vector_db)
