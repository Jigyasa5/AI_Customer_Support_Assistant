import os
from dotenv import load_dotenv
from google import genai

# Load Gemini Client

def load_generator():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env"
        )

    client = genai.Client(
        api_key=api_key
    )

    return client


# Retrieve Context

def retrieve_context(
    query,
    collection,
    embedding_model,
    top_k=3
):

    query_embedding = (
        embedding_model
        .encode([query])
        .tolist()
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results["documents"][0]

    context = "\n\n".join(documents)

    return context

   


def create_prompt(
    query,
    context
):

    return f"""
You are an AI customer support assistant.

IMPORTANT RULES:

1. Answer ONLY using the CONTEXT.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT make up information.
5. If the answer is not present in the context, say exactly:

I don't have enough information to answer this question.

Keep the answer concise.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""
# Generate Answer using Gemini

def generate_answer(
    query,
    collection,
    embedding_model,
    client,
    top_k=3
):

    context = retrieve_context(
        query,
        collection,
        embedding_model,
        top_k
    )

    prompt = create_prompt(
        query,
        context
    )

     # Call Gemini API

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()
    