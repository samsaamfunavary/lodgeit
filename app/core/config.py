import os
from dotenv import load_dotenv

load_dotenv() 
 

class CONFIG:
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    # Database Configuration
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = int(os.environ.get("DB_PORT", 3306))
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_NAME = os.environ.get("DB_NAME", "lodgeit_help_guide")
    
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
    JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # JWT Encryption Configuration
    JWT_ENCRYPTION_KEY = os.environ.get("JWT_ENCRYPTION_KEY", "your-32-character-encryption-key-here")
    JWT_ENCRYPTION_ALGORITHM = "AES-256-GCM"
    
    # Azure OpenAI (single source of truth)
    AZURE_OPEN_API_ENDPOINT = os.environ.get("AZURE_OPEN_API_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

    # OpenAI Configuration (your working setup)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEFAULT_OPENAI_MODEL = os.environ.get("DEFAULT_OPENAI_MODEL")

    # Google Gemini Configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_KEY")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    # Azure AI Search (hardcoded like your working project)
    AZURE_KEY = os.environ.get("AZURE_API_KEY")
    AZURE_ENDPOINT = "https://taxgeniiaisearch.search.windows.net"
    AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "lodgeit-help-guides")

    OPENAI_MODEL = "gpt-4-1106-preview"
    ROLE_USER = "user"
    ROLE_SYSTEM = "system"
    ROLE_ASSISTANT = "assistant"
    ROLE_KNOWLEDGE = "knowledge"
    
    # UI configuration for images in streamed responses
    IMAGE_MAX_WIDTH_PX = int(os.environ.get("IMAGE_MAX_WIDTH_PX", "600"))
