from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def embed_text(text: str):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)

    return response.data[0].embedding


def build_embeddings(chunks):
    vector_db = []

    for i, chunk in enumerate(chunks):

        source = chunk["metadata"]["source"] or "Unknown Source"
        header = chunk["metadata"]["header"] or "Unknown Header"

        embedding = embed_text(
            "Source: " + source + " - " + "Header: " + header + "\n" + chunk["text"]
        )

        vector_db.append(
            {
                "id": i,
                "embedding": embedding,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
        )

    return vector_db
