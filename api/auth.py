import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jose import jwt
from pwdlib import PasswordHash

load_dotenv()

# Password hashing
password_hash = PasswordHash.recommended()

# JWT settings
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "development-secret-key"
)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# PASSWORD HASHING

def hash_password(password):

    return password_hash.hash(password)


def verify_password(
    plain_password,
    hashed_password
):

    return password_hash.verify(
        plain_password,
        hashed_password
    )


# JWT TOKEN

def create_access_token(username):
    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": username,
        "exp": expire
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token