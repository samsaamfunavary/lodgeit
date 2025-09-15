# Authentication System for LodgeIt Help Guides API

This document describes the authentication system implemented for the LodgeIt Help Guides FastAPI application.

## Overview

The authentication system provides:
- User registration and login
- JWT token-based authentication
- Password hashing with bcrypt
- MySQL database integration
- Input validation and security

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `env_template.txt` to `.env` and update the values:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_NAME=lodgeit_help_guide

# JWT Configuration
JWT_SECRET_KEY=your_super_secret_jwt_key_here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3. Initialize Database
```bash
python init_db.py
```

### 4. Start the Application
```bash
python main.py
```

## API Endpoints

### Authentication Endpoints

#### 1. Register User
**POST** `/api/v1/auth/register`

**Request Body:**
```json
{
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123"
}
```

**Response:**
```json
{
    "id": 1,
    "username": "testuser",
    "email": "test@example.com",
    "created_at": "2024-01-01T00:00:00"
}
```

#### 2. Login User
**POST** `/api/v1/auth/login`

**Request Body:**
```json
{
    "username": "testuser",
    "password": "TestPassword123"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
}
```

#### 3. Get Current User
**GET** `/api/v1/auth/me`

**Headers:**
```
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
    "id": 1,
    "username": "testuser",
    "email": "test@example.com",
    "created_at": "2024-01-01T00:00:00"
}
```

#### 4. Logout User
**POST** `/api/v1/auth/logout`

**Response:**
```json
{
    "message": "Successfully logged out"
}
```

## Security Features

### Password Requirements
- Minimum 8 characters
- At least one letter
- At least one number

### Email Validation
- Standard email format validation using regex

### JWT Token Security
- Tokens expire after 30 minutes (configurable)
- Uses HS256 algorithm
- Secret key should be changed in production

### Password Hashing
- Uses bcrypt for secure password hashing
- Salt rounds are automatically handled

## Testing

### Run Authentication Tests
```bash
python test_auth.py
```

### Manual Testing with curl

#### Register a user:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "testuser",
       "email": "test@example.com",
       "password": "TestPassword123"
     }'
```

#### Login:
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "testuser",
       "password": "TestPassword123"
     }'
```

#### Get current user (replace TOKEN with actual token):
```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
     -H "Authorization: Bearer TOKEN"
```

## Integration with Chat API

The authentication system is designed to work alongside the existing chat API. You can:

1. Register/login users
2. Use JWT tokens to authenticate chat requests
3. Track user interactions and history (future enhancement)

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (invalid credentials)
- `500`: Internal Server Error

## Production Considerations

1. **Change JWT Secret Key**: Use a strong, random secret key
2. **Database Security**: Use connection pooling and SSL
3. **Rate Limiting**: Implement rate limiting for auth endpoints
4. **HTTPS**: Always use HTTPS in production
5. **CORS**: Configure CORS properly for your frontend domain
6. **Password Policy**: Consider implementing stronger password requirements
7. **Account Lockout**: Implement account lockout after failed attempts

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check MySQL is running
   - Verify database credentials in `.env`
   - Ensure database exists

2. **JWT Token Error**
   - Check JWT_SECRET_KEY is set
   - Verify token format in Authorization header

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path and virtual environment

### Debug Mode
Set `FLASK_DEBUG=true` in `.env` for detailed error messages and SQL query logging.
