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
    # Docker Subscription
    "What subscription plans does Docker offer?",
    "How can an organization manage Docker subscriptions?",
    "What features are included in Docker Team plans?",
    "What features are included in Docker Business plans?",
    "How do I add members to a Docker organization?",
    "How can administrators manage user access in Docker?",
    "What happens when a Docker subscription expires?",
    "How can an organization review its Docker billing information?",
    # FastAPI Deployment
    "What are the recommended options for deploying a FastAPI application?",
    "What ASGI server can be used to run FastAPI in production?",
    "What is the role of Uvicorn in FastAPI deployment?",
    "Why should FastAPI applications use multiple worker processes?",
    "How can HTTPS be configured for a FastAPI application?",
    "What is the difference between development and production deployment in FastAPI?",
    "How can FastAPI be deployed using Docker?",
    "What considerations are important when deploying FastAPI behind a reverse proxy?",
    "How should application startup and shutdown events be handled in production?",
    # Streamlit Deployment
    "How do I deploy a Streamlit app to Community Cloud?",
    "What repository providers are supported by Streamlit Community Cloud?",
    "What files are required before deploying a Streamlit application?",
    "How do I update a deployed Streamlit application?",
    "How can secrets be managed in Streamlit Community Cloud?",
    "What happens when a Streamlit application deployment fails?",
    "How do I specify Python package dependencies for a Streamlit app?",
    "How can I share a deployed Streamlit application with other users?",
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
