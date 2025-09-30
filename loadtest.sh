#!/bin/bash

# --- Configuration ---
# 1. Paste your NEW, valid JWT token here
TOKEN="Z0FBQUFBQm8xb2g0RXkycGdRanNuMkVoMVoweVNxUGxvc2xkNkpDbUdfeGlQR1lxVTNWV2FTN2FDbG03aTkxcXNiRFVubWd1REx6dldvMERWd3FaRTYzWVNOUUt0QUxRYTh2UUp4cmhseFplQzdrTng2NnlGZjV1MVVrNktRYTFuNGlhRElUenQycS1nYWVsVGRpZmY1Qm16NVc5MjhxNUQtNC1PUFZOckwxZ2tLajJXQVpybHR1YXhUVmlvQTBkdTNEWXdEQkFUU2VGUENzRlJuZVhvT2JVelNUanNOYThWSzhpUVk1dVE3Y2xoYkZFR1lWaXZEdz0="

# 2. Set the number of concurrent requests you want to send
CONCURRENT_REQUESTS=10

# 3. Define the question to ask
QUESTION="Tell me about LodgeiT's key features and integrations."

# 4. API URLs
BASE_URL="http://127.0.0.1:8001/api/v1"
NEW_CHAT_URL="$BASE_URL/new-chat"
MESSAGE_URL="$BASE_URL/chat-lg" # Target the LangGraph endpoint

# --- Script Logic ---

echo "--- Step 1: Creating a new chat session ---"

# Create a new chat and extract the chat_id from the JSON response
CHAT_ID=$(curl -s -X POST "$NEW_CHAT_URL" \
-H "Authorization: Bearer $TOKEN" \
| grep -o '"chat_id":[0-9]*' | cut -d':' -f2)

# --- IMPROVED ERROR CHECKING ---
if ! [[ "$CHAT_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: Could not create a new chat session. The CHAT_ID was not a valid number."
    echo "Please check if your JWT token is correct and not expired."
    exit 1
fi

echo "Successfully created Chat ID: $CHAT_ID"
echo ""
echo "--- Step 2: Sending $CONCURRENT_REQUESTS concurrent requests ---"
echo "Each dot represents one completed request."

# Create the JSON payload for the message
PAYLOAD="{\"chat_id\": $CHAT_ID, \"message\": \"$QUESTION\", \"stream\": true}"

# Loop to send concurrent requests in the background
for i in $(seq 1 $CONCURRENT_REQUESTS)
do
    # Use -o /dev/null to discard the output and -w "." to print the status code
    curl -s -o /dev/null -w "." -X POST "$MESSAGE_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$PAYLOAD" &
done

# Wait for all background jobs to finish
wait

echo ""
echo ""
echo "--- Load test complete! ---"
echo "Check your FastAPI server logs to see how it handled the concurrent requests."

