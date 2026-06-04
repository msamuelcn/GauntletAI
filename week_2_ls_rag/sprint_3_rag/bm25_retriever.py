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


dense_results = vectorstore.similarity_search(query, k=10)
