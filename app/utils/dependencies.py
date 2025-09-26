# ```python:Dependencies:app/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.user import User
from app.models.user import TokenBlacklist
from app.services.auth_service import AuthService

auth_service = AuthService()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Check if token is blacklisted
    is_blacklisted = db.query(TokenBlacklist).filter(TokenBlacklist.token == token).first()
    if is_blacklisted:
        raise credentials_exception

    payload = auth_service.verify_encrypted_token(token)
    if payload is None or "sub" not in payload:
        raise credentials_exception
    
    username: str = payload.get("sub")
    user = auth_service.get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user