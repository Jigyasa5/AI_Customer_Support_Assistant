from fastapi import FastAPI
from pydantic import BaseModel

from src.embeddings import load_embedding_models
from src.vector_database import (
    create_chroma_client,
    create_collection
)
from src.rag import (
    load_generator,
    generate_answer
)


app = FastAPI(
    title="AI Customer Support Assistant",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


# Load embedding model
embedding_models = load_embedding_models()

embedding_model = embedding_models["MiniLM"]


# Load ChromaDB
client = create_chroma_client("./chroma_db")

collection = create_collection(
    client,
    "customer_support"
)


# Load Gemini
generator = load_generator()


@app.get("/")
def home():

    return {
        "message": "AI Customer Support API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "chunks": collection.count()
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:

        return {
            "status": "error",
            "message": "Question cannot be empty."
        }

    answer = generate_answer(
        question,
        collection,
        embedding_model,
        generator,
        top_k=3
    )

    return {
        "status": "success",
        "question": question,
        "answer": answer
    }