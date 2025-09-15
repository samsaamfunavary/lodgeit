# LodgeIt Help Guides Chat API

A simple FastAPI application that provides RAG-powered chat using Azure Search and OpenAI, with streaming responses and proper markdown formatting.

## Features

- **RAG Chat**: Combines Azure Search with OpenAI for intelligent responses
- **Hierarchy Filtering**: Narrow down search results using hierarchy filters
- **Streaming Responses**: Real-time streaming chat responses
- **Markdown Support**: Proper markdown formatting with clickable reference links
- **No Database Dependencies**: Simple and lightweight

## Setup

1. **Activate Virtual Environment**:
   ```bash
   source lodgeit_env/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file with your Azure and OpenAI credentials:
   ```env
   AZURE_OPEN_API_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   OPENAI_API_KEY=your_openai_key
   AZURE_API_KEY=your_azure_search_key
   ```

## Usage

### Start the Server
```bash
python start_server.py
```

The server will start on `http://localhost:8000`

### API Endpoints

1. **POST /api/v1/chat** - Non-streaming chat
   ```json
   {
     "message": "Your question here",
     "hierarchy_filters": ["filter1", "filter2"],
     "index_name": "ato_complete_data2",
     "limit": 3
   }
   ```

2. **POST /api/v1/chat/stream** - Streaming chat
   ```json
   {
     "message": "Your question here",
     "hierarchy_filters": ["filter1", "filter2"],
     "index_name": "ato_complete_data2",
     "limit": 3
   }
   ```

3. **GET /api/v1/chat/stream** - Streaming chat via GET
   ```
   /api/v1/chat/stream?message=your_question&hierarchy_filters=filter1,filter2&index_name=ato_complete_data2&limit=3
   ```

### Test the API

1. Open `test_chat.html` in your browser
2. Enter your question
3. Add hierarchy filters (comma-separated)
4. Click "Start Streaming Chat" for real-time responses

## How It Works

1. **User Input**: Receives chat message and hierarchy filters
2. **RAG Search**: Uses Azure Search to find relevant documents based on filters
3. **Context Building**: Creates a prompt with relevant document context
4. **LLM Response**: OpenAI generates response using the context
5. **Streaming**: Response is streamed back to the frontend in real-time
6. **Markdown Rendering**: Frontend renders markdown with clickable links

## Response Format

The API returns:
- **LLM Response**: AI-generated answer in markdown format
- **Relevant Documents**: Source documents with titles, hierarchy, content, and URLs
- **Clickable References**: Markdown links to source documents

## Example Response

```markdown
Based on the provided context, here's how to configure the system:

1. **Initial Setup**: Follow the configuration steps in [Document 1](url1)
2. **Hierarchy Configuration**: Use the settings described in [Document 2](url2)

The system requires proper authentication as outlined in the documentation.
```

## Troubleshooting

- **Azure Search Issues**: Check your Azure Search endpoint and API key
- **OpenAI Issues**: Verify your OpenAI API key and model configuration
- **Streaming Issues**: Ensure your browser supports Server-Sent Events (SSE)

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
