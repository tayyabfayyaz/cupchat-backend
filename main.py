import os
import json
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
import logging
from typing import List, Optional
from datetime import datetime
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

logger.info("=== STARTING AI AGENT API SERVER ===")

# Initialize Supabase
def initialize_supabase():
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("❌ Supabase credentials missing")
        return None
    
    try:
        logger.info("🔄 Connecting to Supabase...")
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase connected successfully!")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return None

supabase = initialize_supabase()

# Custom Gemini AI implementation (replaces agents SDK)
class GeminiAIClient:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
    async def generate_content(self, prompt: str, instructions: str = ""):
        if not self.api_key:
            raise Exception("GEMINI_API_KEY not configured")
        
        full_prompt = f"{instructions}\n\nUser: {prompt}\nAssistant:" if instructions else prompt
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{
                                "text": full_prompt
                            }]
                        }]
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"Gemini AI request failed: {e}")
            raise Exception(f"AI service unavailable: {str(e)}")

# Initialize Gemini client
gemini_client = GeminiAIClient()

app = FastAPI(title="AI Agent API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class AgentCreate(BaseModel):
    name: str
    instructions: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    agent_name: str
    agent_id: str

# Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    
    agent_data = get_agent_by_api_key(api_key.strip())
    if not agent_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return agent_data

# Database Operations
def save_agent(agent_data):
    if not supabase:
        raise Exception("Supabase not configured")
    
    try:
        supabase_data = {
            "id": agent_data["id"],
            "name": agent_data["name"], 
            "instructions": agent_data["instructions"],
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.from_("agents").insert(supabase_data).execute()
        
        if hasattr(response, 'error') and response.error:
            error_msg = f"Supabase error: {response.error.message}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        logger.info(f"✅ Agent '{agent_data['name']}' saved to Supabase!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save agent to Supabase: {e}")
        raise Exception(f"Failed to save agent: {e}")

def save_api_key(api_key_data):
    if not supabase:
        raise Exception("Supabase not configured")
    
    try:
        supabase_data = {
            "api_key": api_key_data["api_key"],
            "agent_id": api_key_data["agent_id"],
            "name": api_key_data["name"],
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.from_("api_keys").insert(supabase_data).execute()
        
        if hasattr(response, 'error') and response.error:
            error_msg = f"Supabase error: {response.error.message}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        logger.info(f"✅ API key saved to Supabase for agent '{api_key_data['name']}'!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save API key to Supabase: {e}")
        raise Exception(f"Failed to save API key: {e}")

def get_agent_by_api_key(api_key):
    if not supabase:
        return None
    
    try:
        # First get the API key record
        api_key_response = supabase.from_("api_keys").select("*").eq("api_key", api_key).execute()
        
        if not api_key_response.data or len(api_key_response.data) == 0:
            return None
        
        api_key_data = api_key_response.data[0]
        agent_id = api_key_data["agent_id"]
        
        # Then get the agent data
        agent_response = supabase.from_("agents").select("*").eq("id", agent_id).execute()
        
        if not agent_response.data or len(agent_response.data) == 0:
            return None
        
        agent_data = agent_response.data[0]
        
        # Combine both data
        return {
            "agent_id": agent_data["id"],
            "agent_name": agent_data["name"],
            "instructions": agent_data["instructions"],
            "api_key_data": api_key_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching agent by API key: {e}")
        return None

# API Routes
@app.get("/")
async def root():
    return {
        "message": "AI Agent API Server is running",
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/agents")
async def create_agent(agent: AgentCreate):
    try:
        logger.info(f"🎯 Creating new agent: {agent.name}")
        
        # Validate input
        if not agent.name.strip() or not agent.instructions.strip():
            raise HTTPException(status_code=400, detail="Name and instructions are required")
        
        # Generate unique IDs
        agent_id = str(uuid.uuid4())
        api_key = str(uuid.uuid4()).replace("-", "")
        
        # Prepare data
        new_agent = {
            "id": agent_id,
            "name": agent.name.strip(),
            "instructions": agent.instructions.strip()
        }
        
        new_api_key = {
            "api_key": api_key,
            "agent_id": agent_id,
            "name": agent.name.strip()
        }
        
        # Save to database
        save_agent(new_agent)
        save_api_key(new_api_key)
        
        logger.info(f"✅ Agent '{agent.name}' created successfully!")
        
        # Return response
        return {
            "success": True,
            "apiKey": api_key,
            "agentId": agent_id,
            "apiUrl": "https://your-railway-url.up.railway.app/api/chat",  # Update this after deployment
            "message": f"Agent '{agent.name}' created successfully",
            "agent": {
                "name": agent.name,
                "id": agent_id
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(chat: ChatRequest, agent_data: dict = Depends(verify_api_key)):
    try:
        logger.info(f"💬 Chat request for agent: {agent_data['agent_name']}")
        
        # Validate message
        if not chat.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get agent instructions from Supabase data
        agent_name = agent_data["agent_name"]
        instructions = agent_data["instructions"]
        agent_id = agent_data["agent_id"]
        
        # Process with Gemini AI
        ai_response = await gemini_client.generate_content(chat.message.strip(), instructions)
        
        logger.info(f"✅ Chat completed successfully for agent '{agent_name}'")
        
        return ChatResponse(
            reply=ai_response,
            agent_name=agent_name,
            agent_id=agent_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "supabase_connected": supabase is not None,
        "gemini_configured": os.getenv('GEMINI_API_KEY') is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)