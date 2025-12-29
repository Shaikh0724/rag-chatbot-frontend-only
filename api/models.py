from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

# User Registration ke liye
class UserSignup(BaseModel):
    email: EmailStr
    password: str

# Login ke liye
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatReq(BaseModel):
    query: str
    model_name: str = "gpt"
    session_id: Optional[str] = None
    user_email: str

# Chat Message ka structure
class Message(BaseModel):
    role: str  # "user" ya "assistant"
    content: str
    timestamp: datetime = datetime.now()

# Poori Chat Session ka structure
class ChatSession(BaseModel):
    id: Optional[str] = None
    user_email: str
    title: str
    messages: List[Message]