import os
from io import BytesIO
from typing import List, Optional
from docx import Document
from pypdf import PdfReader
import io
from docx import Document as DocxDocument  # Word ke liye alag naam
from langchain_core.documents import Document as LangchainDocument  # Langchain ke liye alag
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from bson import ObjectId
from datetime import datetime
from api.models import UserSignup, UserLogin, ChatReq
from api.auth import verify_access_token
# RAG & AI Imports
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
from google import genai

# Database & Auth Imports
from api.database import db, hash_pass, verify_pass, create_token as create_access_token 
app = FastAPI()

# Frontend Connect: static folder ko serve karne ke liye
#app.mount("/frontend", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Clients Setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "AbdulWahab")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# --- Pydantic Models ---
class UserAuth(BaseModel):
    email: str
    password: str

class ChatReq(BaseModel):
    query: str
    model_name: str = "gpt"
    session_id: Optional[str] = None
    user_email: str

# --- 1. Auth Endpoints ---
@app.post("/signup")
async def signup(user: UserAuth):
    if await db.users.find_one({"email": user.email}):
        raise HTTPException(400, "Email already exists")
    await db.users.insert_one({"email": user.email, "password": hash_pass(user.password)})
    return {"status": "success"}

@app.post("/login")
async def login(user: UserAuth):
    db_user = await db.users.find_one({"email": user.email})
    if not db_user or not verify_pass(user.password, db_user["password"]):
        raise HTTPException(400, "Invalid credentials")
    return {"token": create_access_token({"sub": user.email}), "email": user.email}

# --- 2. Indexing Endpoint ---
@app.post("/index")
async def index_doc(file: UploadFile = File(...)):
    try:
        data = await file.read()
        text = ""
        filename = file.filename.lower()

        # --- YAHAN NAYA LOGIC ADD HOGA ---
        if filename.endswith(".pdf"):
            pdf = PdfReader(BytesIO(data))
            for page in pdf.pages: 
                text += page.extract_text() or ""
        
        elif filename.endswith(".docx"):
            doc = DocxDocument(io.BytesIO(data))
            for para in doc.paragraphs:
                text += para.text + "\n"
            
        elif filename.endswith(".txt"):
            text = data.decode("utf-8")

        
        else:
            raise HTTPException(400, "Sirf PDF, DOCX, aur TXT support hain.")
        # --------------------------------

        if not text.strip():
            raise HTTPException(400, "File khali hai.")

        # --- YAHAN SE AAPKA QDRANT WALA PURANA LOGIC SHURU HOGA ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents([LangchainDocument(page_content=text, metadata={"source": file.filename})])

        QdrantVectorStore.from_documents(
            documents=chunks, embedding=embeddings,
            url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=COLLECTION
        )
        return {"status": "success", "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(500, str(e))

# --- 3. Chat & History Endpoint ---
@app.post("/search")
async def chat(req: ChatReq):
    try:
        # RAG Retrieval
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
        docs = vectorstore.similarity_search(req.query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Generation
        prompt = f"Context: {context}\n\nQuestion: {req.query}"
        if req.model_name == "gemini":
            res = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            answer = res.text

        elif req.model_name == "exai":
            from openai import OpenAI as XAIClient
            x_client = XAIClient(
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1",
            )
            res = x_client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}]
            )    
            answer = res.choices[0].message.content
        else:
            res = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
            answer = res.choices[0].message.content

        # Save to MongoDB History
        chat_msg = {"user": req.query, "bot": answer, "time": datetime.utcnow()}
        if req.session_id:
            await db.sessions.update_one({"_id": ObjectId(req.session_id)}, {"$push": {"messages": chat_msg}})
            session_id = req.session_id
        else:
            new_s = await db.sessions.insert_one({
                "user_email": req.user_email,
                "title": req.query[:30] + "...",
                "messages": [chat_msg]
            })
            session_id = str(new_s.inserted_id)

        return {"response": answer, "session_id": session_id, "sources": list(set([d.metadata['source'] for d in docs]))}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/history/{email}")
async def get_history(email: str):
    sessions = await db.sessions.find({"user_email": email}).to_list(100)
    for s in sessions: s["_id"] = str(s["_id"])
    return sessions








'''
import os
from io import BytesIO
from dotenv import load_dotenv

# 1. Environment Variables Load Karein
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader

# --- Latest RAG Imports (No more init_from issues) ---
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "AbdulWahab")

# Clients Setup
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

@app.get("/")
async def serve_home():
    # Frontend serving
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Server is up, but index.html missing in root."}

@app.post("/index")
async def index_doc(file: UploadFile = File(...)):
    try:
        # 1. Extraction
        data = await file.read()
        text = ""
        if file.filename.lower().endswith(".pdf"):
            pdf = PdfReader(BytesIO(data))
            for page in pdf.pages:
                text += page.extract_text() or ""
        else:
            text = data.decode("utf-8")

        # 2. Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents([Document(page_content=text, metadata={"source": file.filename})])

        # 3. CORRECT INDEXING (Using Latest QdrantVectorStore)
        #
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION,
            force_recreate=False
        )
        
        print(f"DEBUG: Successfully indexed {file.filename}")
        return {"status": "success", "message": f"File {file.filename} indexed!"}

    except Exception as e:
        print(f"INDEXING FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatReq(BaseModel):
    query: str
    model_name: str = "gpt"

@app.post("/search")
async def chat(req: ChatReq):
    try:
        # Connection for search using QdrantClient directly to avoid confusion
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        vectorstore = QdrantVectorStore(
            client=client, 
            collection_name=COLLECTION, 
            embedding=embeddings
        )
        
        docs = vectorstore.similarity_search(req.query, k=3)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Use this context: {context}\n\nQuestion: {req.query}"

        if req.model_name == "gemini":
            res = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            answer = res.text
        else:
            res = openai_client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}]
            )
            answer = res.choices[0].message.content

        return {"response": answer, "sources": list(set([d.metadata['source'] for d in docs]))}
    except Exception as e:
        print(f"SEARCH FAILED: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


'''

