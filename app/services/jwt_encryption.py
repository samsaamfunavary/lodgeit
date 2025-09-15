"""
JWT Encryption Service for additional security
Encrypts JWT tokens before sending to frontend
"""

import base64
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from app.core.config import CONFIG

class JWTEncryptionService:  
    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key from environment or generate new one"""
        encryption_key_str = CONFIG.JWT_ENCRYPTION_KEY
        
        if encryption_key_str == "your-32-character-encryption-key-here":
            # Generate a new key if not set
            return Fernet.generate_key()
        
        # Convert string key to bytes using PBKDF2
        password = encryption_key_str.encode()
        salt = b'lodgeit_jwt_salt'  # Fixed salt for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_jwt(self, jwt_token: str) -> str:
        """Encrypt JWT token before sending to frontend"""
        try:
            # Convert JWT string to bytes
            jwt_bytes = jwt_token.encode('utf-8')
            
            # Encrypt the JWT
            encrypted_jwt = self.cipher.encrypt(jwt_bytes)
            
            # Encode to base64 for safe transmission
            encrypted_b64 = base64.urlsafe_b64encode(encrypted_jwt).decode('utf-8')
            
            return encrypted_b64
        except Exception as e:
            raise Exception(f"JWT encryption failed: {str(e)}")
    
    def decrypt_jwt(self, encrypted_jwt: str) -> str:
        """Decrypt JWT token received from frontend"""
        try:
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_jwt.encode('utf-8'))
            
            # Decrypt the JWT
            decrypted_jwt = self.cipher.decrypt(encrypted_bytes)
            
            # Convert back to string
            jwt_token = decrypted_jwt.decode('utf-8')
            
            return jwt_token
        except Exception as e:
            raise Exception(f"JWT decryption failed: {str(e)}")
    
    def generate_encryption_key(self) -> str:
        """Generate a new encryption key for environment setup"""
        return Fernet.generate_key().decode('utf-8')

# Create global instance
jwt_encryption = JWTEncryptionService()
