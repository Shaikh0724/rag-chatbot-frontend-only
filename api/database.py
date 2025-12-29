import os
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from dotenv import load_dotenv

# Config
SECRET_KEY = "SUPER_SECRET_KEY_XYZ" # Isay .env mein rakhein
ALGORITHM = "HS256"
load_dotenv()

# MongoDB Connection
client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
db = client.chatbot_db

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_pass(password): return pwd_context.hash(password)
def verify_pass(plain, hashed): return pwd_context.verify(plain, hashed)

def create_token(data: dict):
    return jwt.encode({**data, "exp": datetime.utcnow() + timedelta(days=1)}, SECRET_KEY, algorithm=ALGORITHM)