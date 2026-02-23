from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Portfolio Chat API")

# CORS - allow your portfolio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ─── MODELS ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    profile: str

class ChatResponse(BaseModel):
    reply: str

SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant representing a developer. Your sole purpose is to showcase the developer's 
capabilities, skills, and experience based on the profile provided.

Guidelines:
1. Be professional, concise, and persuasive.
2. Focus strictly on how the developer can solve the visitor's problems.
3. Do not engage in casual conversation or off-topic discussion.
4. Only answer based on the provided profile.

Developer Profile:
{profile}
"""

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Portfolio Chat API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        prompt = f"{SYSTEM_PROMPT_TEMPLATE.format(profile=request.profile)}\n\nVisitor says: {request.message}"
        print(prompt)
        response = model.generate_content(prompt)
        return ChatResponse(reply=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")