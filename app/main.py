from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router

app = FastAPI(
    title="LodgeIt Help Guides Chat API",
    description="A FastAPI application for RAG-powered chat using Azure Search and OpenAI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*","X-Reference-Documents"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to LodgeIt Help Guides Chat API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/v1/chat",
            "chat_stream": "/api/v1/chat/stream",
            "chat_stream_get": "/api/v1/chat/stream?message=your_question&hierarchy_filters=filter1,filter2",
            "auth_register": "/api/v1/auth/register",
            "auth_login": "/api/v1/auth/login",
            "auth_me": "/api/v1/auth/me",
            "auth_logout": "/api/v1/auth/logout"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
