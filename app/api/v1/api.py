from fastapi import APIRouter
from app.api.v1.endpoints import chat, auth

api_router = APIRouter()
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
