from datetime import UTC, datetime, timedelta
from typing import Optional
import warnings
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.models.user import User
from app.core.config import CONFIG
from app.services.jwt_encryption import jwt_encryption

# Suppress bcrypt version warning
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.secret_key = CONFIG.JWT_SECRET_KEY
        self.algorithm = CONFIG.JWT_ALGORITHM
        self.access_token_expire_minutes = CONFIG.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_encrypted_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create an encrypted JWT access token for extra security"""
        # First create the regular JWT
        jwt_token = self.create_access_token(data, expires_delta)
        
        # Then encrypt it
        encrypted_jwt = jwt_encryption.encrypt_jwt(jwt_token)
        return encrypted_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
    
    def verify_encrypted_token(self, encrypted_token: str) -> Optional[dict]:
        """Verify and decode an encrypted JWT token"""
        try:
            # First decrypt the token
            jwt_token = jwt_encryption.decrypt_jwt(encrypted_token)
            
            # Then verify the JWT
            return self.verify_token(jwt_token)
        except Exception:
            return None
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return None
        if not self.verify_password(password, user.password_hash):
            return None
        return user
    
    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    def create_user(self, db: Session, username: str, email: str, password: str) -> User:
        """Create a new user"""
        hashed_password = self.get_password_hash(password)
        user = User(
            username=username,
            email=email,
            password_hash=hashed_password
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def logout_user(self, db: Session, token: str) -> bool:
        """
        Invalidate a JWT token by adding it to a token blacklist.
        Returns True if the token was successfully blacklisted.
        """
        # Assuming you have a TokenBlacklist model/table
        from app.models import TokenBlacklist

        # Check if token is already blacklisted
        existing = db.query(TokenBlacklist).filter(TokenBlacklist.token == token).first()
        if existing:
            return True  # Already blacklisted

        blacklisted_token = TokenBlacklist(token=token)
        db.add(blacklisted_token)
        db.commit()
        return True