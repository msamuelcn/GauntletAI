import numpy as np

from embeddings import embeddings


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, vector_db, top_k=5):
    query_embedding = embeddings.embed_text(query)

    scored = []

    for item in vector_db:
        score = cosine_similarity(query_embedding, item["embedding"])

        scored.append(
            {
                "score": score,
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]
