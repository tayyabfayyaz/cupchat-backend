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

# Gemini AI imports
from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
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
        
        # Test connection
        result = client.from_("agents").select("id").limit(1).execute()
        logger.info("✅ Supabase connected successfully!")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return None

supabase = initialize_supabase()

# Initialize Gemini AI
def initialize_gemini():
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not found in environment variables")
        return None
    
    try:
        logger.info("🔄 Initializing Gemini AI...")
        
        # Create external client for Gemini
        external_client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Create model configuration
        model = OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=external_client,
        )
        
        # Create run configuration
        config = RunConfig(
            model=model,
            model_provider=external_client,
            tracing_disabled=True,
        )
        
        logger.info("✅ Gemini AI initialized successfully!")
        return config
        
    except Exception as e:
        logger.error(f"❌ Gemini AI initialization failed: {e}")
        return None

gemini_config = initialize_gemini()

app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    description="Production-ready AI Agent API with Gemini AI and Supabase",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://your-production-domain.com"  # Add your production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
    """Verify API key and return agent data"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    
    agent_data = get_agent_by_api_key(api_key.strip())
    if not agent_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return agent_data

# Database Operations
def save_agent(agent_data):
    """Save agent to Supabase with proper error handling"""
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
    """Save API key to Supabase with proper error handling"""
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
    """Get agent data by API key from Supabase"""
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

def get_agent_by_id(agent_id):
    """Get agent data by ID from Supabase"""
    if not supabase:
        return None
    
    try:
        response = supabase.from_("agents").select("*").eq("id", agent_id).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching agent by ID: {e}")
        return None

# AI Chat Function
async def chat_with_gemini(agent_name: str, instructions: str, message: str):
    """Chat with Gemini AI using agent instructions"""
    if not gemini_config:
        raise Exception("Gemini AI not configured")
    
    try:
        # Create agent with instructions from Supabase
        agent = Agent(
            name=agent_name,
            instructions=instructions
        )
        
        # Run the agent with the user's message
        logger.info(f"🤖 Processing message with agent '{agent_name}'")
        result = await Runner.run(agent, message, run_config=gemini_config)
        
        logger.info(f"✅ AI response generated successfully")
        return result.final_output
        
    except Exception as e:
        logger.error(f"❌ Gemini AI processing failed: {e}")
        raise Exception(f"AI processing error: {e}")

# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Agent API Server is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": {
            "ai_provider": "Gemini AI",
            "database": "Supabase",
            "authentication": "API Key"
        }
    }

@app.post("/api/agents")
async def create_agent(agent: AgentCreate):
    """Create a new AI agent with custom instructions"""
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
            "apiUrl": "http://localhost:8000/api/chat",
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
    """Chat with an AI agent using Gemini AI"""
    try:
        logger.info(f"💬 Chat request for agent: {agent_data['agent_name']}")
        logger.info(f"📨 Message: {chat.message[:100]}...")
        
        # Validate message
        if not chat.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get agent instructions from Supabase data
        agent_name = agent_data["agent_name"]
        instructions = agent_data["instructions"]
        agent_id = agent_data["agent_id"]
        
        # Process with Gemini AI
        ai_response = await chat_with_gemini(agent_name, instructions, chat.message.strip())
        
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

@app.get("/api/agent/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get agent information by ID"""
    try:
        agent_data = get_agent_by_id(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Return limited info (no sensitive data)
        return {
            "id": agent_data["id"],
            "name": agent_data["name"],
            "created_at": agent_data.get("created_at")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching agent: {str(e)}")

# Debug and monitoring endpoints
@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check system status"""
    return {
        "supabase_connected": supabase is not None,
        "gemini_configured": gemini_config is not None,
        "environment": os.getenv('ENVIRONMENT', 'development'),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/agents")
async def debug_agents():
    """Debug endpoint to list all agents (for admin only)"""
    if not supabase:
        return {"error": "Supabase not connected"}
    
    try:
        response = supabase.from_("agents").select("id, name, created_at").execute()
        return {
            "agents": response.data if response.data else [],
            "count": len(response.data) if response.data else 0
        }
    except Exception as e:
        return {"error": str(e)}

# Error handlers
# @app.exception_handler(500)
# async def internal_error_handler(request, exc):
#     logger.error(f"Internal server error: {exc}")
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error"}
#     )

# @app.exception_handler(404)
# async def not_found_handler(request, exc):
#     return JSONResponse(
#         status_code=404,
#         content={"detail": "Endpoint not found"}
#     )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI Agent API Server starting up...")
    logger.info(f"📊 Supabase connected: {supabase is not None}")
    logger.info(f"🤖 Gemini AI configured: {gemini_config is not None}")
    logger.info("✅ Server ready to accept requests")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 AI Agent API Server shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    config = {
        "host": os.getenv('HOST', '0.0.0.0'),
        "port": int(os.getenv('PORT', 8000)),
        "reload": os.getenv('ENVIRONMENT') == 'development',
        "workers": int(os.getenv('WORKERS', 1)) if os.getenv('ENVIRONMENT') == 'production' else 1,
        "log_level": "info"
    }
    
    logger.info(f"🚀 Starting server on {config['host']}:{config['port']}")
    logger.info(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info("📚 API documentation: http://localhost:8000/docs")
    
    uvicorn.run("server:app", **config)