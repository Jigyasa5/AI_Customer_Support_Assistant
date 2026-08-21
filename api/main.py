from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt


from src.embeddings import load_embedding_models
from src.vector_database import (
    create_chroma_client,
    create_collection
)
from src.rag import (
    load_generator,
    generate_answer
)
from api.models import (
    create_database,
    create_user,
    get_user,
    save_chat,
    get_chat_history
)

from api.auth import (
    hash_password,
    verify_password,
    create_access_token,
    SECRET_KEY,
    ALGORITHM
)

security = HTTPBearer()

create_database()

app = FastAPI(
    title="AI Customer Support Assistant",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

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


# Get current user
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        if not username:
            raise HTTPException(
                status_code=401,
                detail="Invalid token."
            )

        return username

    except Exception:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token."
        )

    
# Home
@app.get("/")
def home():

    return {
        "message": "AI Customer Support API is running"
    }

# Health 
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "chunks": collection.count()
    }

# Ask question
@app.post("/ask")
def ask_question(request: QuestionRequest,
                username: str = Depends(get_current_user)):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    answer = generate_answer(
        question,
        collection,
        embedding_model,
        generator,
        top_k=3
    )

    save_chat(
        username,
        question,
        answer
    )

    return {
        "status": "success",
        "username": username,
        "question": question,
        "answer": answer
    }

# User registration 
@app.post("/register")
def register(request: RegisterRequest):
    username = request.username.strip()

    if not username or not request.password:
        raise HTTPException(
            status_code=400,
            detail="Username and password are required."
        )

    hashed_password = hash_password(
        request.password
    )

    created = create_user(
        username,
        hashed_password
    )

    if not created:
        raise HTTPException(
            status_code=400,
            detail="Username already exists."
        )

    return {
        "status": "success",
        "message": "User registered successfully."
    }

# User login
@app.post("/login")
def login(request: LoginRequest):
    user = get_user(
        request.username
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    user_id, username, hashed_password = user

    if not verify_password(
        request.password,
        hashed_password
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password."
        )

    token = create_access_token(
        username
    )

    return {
        "status": "success",
        "access_token": token,
        "token_type": "bearer"
    }



# Profile 
@app.get("/profile")
def profile(
    username: str = Depends(get_current_user)
):

    return {
        "status": "success",
        "username": username
    }

# Chat history
@app.get("/history")
def history(
    username: str = Depends(get_current_user)
):

    chats = get_chat_history(username)

    return {
        "status": "success",
        "username": username,
        "history": [
            {
                "question": chat[0],
                "answer": chat[1],
                "created_at": chat[2]
            }
            for chat in chats
        ]
    }