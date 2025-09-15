# Classifier Integration with RAG System

## Overview

The classifier integration automatically determines which RAG index to use based on user queries, eliminating the need for manual index selection. The system intelligently routes queries to the most appropriate knowledge base.

## Architecture

```
User Query → Classifier Service → Index Selection → RAG Agent → Response
```

### Components

1. **ClassifierService** (`app/services/classifier_service.py`)
   - Uses Google Gemini API for query classification
   - Maps queries to appropriate indexes
   - Handles index name mapping

2. **ChatService** (Updated)
   - Integrates classifier for automatic index selection
   - Provides specialized prompts for each index type
   - Maintains backward compatibility

3. **Chat Endpoints** (Updated)
   - Support automatic classification
   - Optional manual index override
   - New `/classify` endpoint for testing

## Index Mapping

| Classifier Output | Azure Search Index | Purpose |
|------------------|-------------------|---------|
| `helpguide` | `lodgeit-help-guides` | Technical how-to guides and troubleshooting |
| `pricing` | `pricing` | Pricing plans and subscription information |
| `taxgenii` | `Taxgenii` | ATO procedures and tax professional guidance |
| `website` | `logit-website` | Product features and general information |

## Usage

### Automatic Classification (Recommended)

```python
# The system will automatically classify the query
response = await chat_service.chat_with_rag(
    message="How much does LodgeiT cost?",
    hierarchy_filters=[],
    index_name=None,  # Auto-classify
    limit=3
)
```

### Manual Index Override

```python
# Override automatic classification
response = await chat_service.chat_with_rag(
    message="How much does LodgeiT cost?",
    hierarchy_filters=[],
    index_name="pricing",  # Force specific index
    limit=3
)
```

### API Endpoints

#### POST `/chat`
- **Description**: Non-streaming chat with automatic classification
- **Body**: 
  ```json
  {
    "message": "Your query here",
    "hierarchy_filters": [],
    "index_name": null,  // Optional override
    "limit": 3
  }
  ```

#### POST `/chat/stream`
- **Description**: Streaming chat with automatic classification
- **Body**: Same as `/chat`

#### POST `/classify`
- **Description**: Test classification without full RAG processing
- **Body**:
  ```json
  {
    "query": "Your query here"
  }
  ```

## Specialized Prompts

Each index has a tailored system prompt:

### LodgeiT Help Guides
- Focus on step-by-step instructions
- Technical troubleshooting
- Software-specific guidance

### Pricing
- Subscription plans and costs
- Feature comparisons
- Plan recommendations

### Taxgenii
- ATO procedures and compliance
- Professional tax practice management
- Regulatory requirements

### LodgeiT Website
- Product capabilities and features
- Target user information
- Integration details

## Testing

### Run Integration Test
```bash
python test_classifier_integration.py
```

### Test Individual Classification
```bash
curl -X POST "http://localhost:8000/api/v1/chat/classify" \
  -H "Content-Type: application/json" \
  -d '{"query": "How much does LodgeiT cost?"}'
```

### Test Full RAG with Classification
```bash
curl -X POST "http://localhost:8000/api/v1/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I connect to QuickBooks?"}'
```

## Configuration

The classifier uses the following environment variables:
- `GEMINI_API_KEY`: Google Gemini API key
- `GEMINI_MODEL`: Gemini model name (default: gemini-1.5-flash)
- `GEMINI_BASE_URL`: Gemini API base URL

## Error Handling

- **Classification Failure**: Falls back to `lodgeit-help-guides` index
- **API Errors**: Graceful degradation with error messages
- **Invalid Queries**: Proper validation and error responses

## Benefits

1. **Automatic Routing**: No manual index selection required
2. **Improved Accuracy**: Specialized prompts for each knowledge domain
3. **Better User Experience**: Seamless query processing
4. **Maintainability**: Centralized classification logic
5. **Flexibility**: Optional manual override when needed

## Migration Notes

- Existing API calls continue to work unchanged
- `index_name` parameter is now optional
- New `classified_index` field in responses
- Backward compatibility maintained

## Monitoring

The system provides:
- Classification results in API responses
- Index mapping information
- Error logging for failed classifications
- Performance metrics through existing logging

## Future Enhancements

- Confidence scoring for classifications
- Multi-index query support
- Learning from user feedback
- Advanced query preprocessing



