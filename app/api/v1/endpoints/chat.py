import pprint
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session,joinedload
import json
from typing import Optional, List

# Local Imports
from app.services import chat_healpers
from app.services.chat_service import ChatService
from app.utils.dependencies import get_db, get_current_user
from app.schemas.chat import ChatMessageResponse, ChatRequestWidget, ChatSessionInfo, NewChatRequest, NewChatResponse, ChatRequest
from app.models.user import User
from app.models.chat import Chat, ChatMessage, MessageSource
from app.services.chat_service_lg import LangGraphChatService
from app.services.chat_healpers import add_message_to_db, add_sources_to_message_in_db, create_chat_session ,get_chat_history_for_session,get_chat_session_for_user# Or wherever you placed this function


router = APIRouter()
lg_chat_service = LangGraphChatService() # The new LangGraph service

chat_service = ChatService()


# --- API ENDPOINTS ---

@router.post("/new-chat", response_model=NewChatResponse)
async def create_new_chat(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Creates a new, empty chat session for the authenticated user.
    """
    try:
        # Create the chat session with a default title
        chat_session = create_chat_session(db, current_user)
        
        # We no longer save an initial message here.
        
        return NewChatResponse(chat_id=chat_session.id, title=chat_session.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create new chat: {e}")

@router.post("/chat")
async def send_chat_message(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Handles sending a message to an EXISTING chat session and saves the conversation.
    Updates the chat title if it's the first message.
    """
    try:
        # 1. Get the chat session, which also verifies ownership
        chat_session = get_chat_session_for_user(db, chat_request.chat_id, current_user)
        
        # 2. Update the chat title if it's the first message
        if chat_session.title == "New Chat":
            chat_session.title = chat_request.message[:50]
            db.commit()
        
        # 3. Save the user's incoming message to the database
        add_message_to_db(db, chat_session.id, "user", chat_request.message)

        # 4. Handle streaming vs. non-streaming response generation
        if chat_request.stream:
            # --- STREAMING LOGIC ---
            async def generate_stream():
                full_response_text = ""
                final_references = []

                prep_data = chat_service.prepare_rag_for_streaming(
                    message=chat_request.message,
                    hierarchy_filters=chat_request.hierarchy_filters or [],
                    index_name=chat_request.index_name,
                    limit=chat_request.limit
                )
                final_references = prep_data.get("relevant_documents", [])
                
                streamer = None
                if prep_data.get("classified_index") == "ato_complete_data2":
                    streamer = chat_service.chat_with_taxgenii_streaming(message=chat_request.message)
                else:
                    messages = prep_data.get("messages", [])
                    streamer = chat_service.chat_with_rag_streaming(messages=messages)
                
                # Process the stream from either source
                async for event in streamer:
                    chunk = ""
                    if isinstance(event, dict): # From TaxGenii
                        if event["type"] == "content":
                            chunk = event["data"]
                        elif event["type"] == "references":
                            final_references = event["data"] # Update references from stream
                    else: # It's a raw string chunk from OpenAI
                        chunk = event
                    
                    if chunk:
                        full_response_text += chunk
                        # Yield each chunk in the Server-Sent Event (SSE) format, wrapped in JSON
                        yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"
                
                # After the stream is complete, save the assistant's full response to the database
                if full_response_text:
                    assistant_message = add_message_to_db(db, chat_session.id, "assistant", full_response_text)
                    if final_references:
                        add_sources_to_message_in_db(db, assistant_message.id, final_references)

                # Now, send the collected references to the frontend as a structured message
                references_data = {"type": "references", "data": final_references}
                yield f"data: {json.dumps(references_data)}\n\n"

                # Finally, send a signal that the stream is complete
                done_data = {"type": "done"}
                yield f"data: {json.dumps(done_data)}\n\n"

            headers = {"X-Chat-Id": str(chat_session.id), "Access-Control-Expose-Headers": "X-Chat-Id"}
            return StreamingResponse(generate_stream(), media_type="text/event-stream", headers=headers)

        else:
            # --- NON-STREAMING LOGIC ---
            response_data = await chat_service.chat_with_rag(
                message=chat_request.message,
                hierarchy_filters=chat_request.hierarchy_filters or [],
                index_name=chat_request.index_name,
                limit=chat_request.limit
            )
            
            assistant_message = add_message_to_db(db, chat_session.id, "assistant", response_data["response"])
            if response_data.get("relevant_documents"):
                add_sources_to_message_in_db(db, assistant_message.id, response_data["relevant_documents"])
            
            response_data["chat_id"] = chat_session.id
            return Response(content=json.dumps(response_data), media_type="application/json")
        
    except HTTPException as e:
        # Re-raise HTTP exceptions directly
        raise e
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")


# Add this new endpoint to your existing chat.py file

@router.post("/chat-widget")
async def send_chat_message_widget(
    chat_request: ChatRequestWidget,
):
    """
    Handles stateless, unauthenticated chat messages for a public website widget.
    The frontend is responsible for sending the entire chat history with each message.
    """
    try:
        if chat_request.stream == True:

                async def generate_stream():
                    # Convert the simple list of dicts from the frontend into LangGraph's message format
                    history = []
                    for msg in chat_request.chat_history:
                        if msg.get("role") == "user":
                            history.append(HumanMessage(content=msg.get("content", "")))
                        elif msg.get("role") == "assistant":
                            history.append(AIMessage(content=msg.get("content", "")))

                    # Prepare the initial state for the LangGraph agent
                    initial_state = {
                        "messages": history,
                        "userInput": chat_request.message,
                        "stream":chat_request.stream
                    }
                    
                    full_response_text = ""
                    final_references = []

                    # Stream the response from the LangGraph agent
                    async for event in lg_chat_service.graph.astream(initial_state):
                        for node_name, state_update in event.items():
                            if state_update is None: continue

                            if "final_response_chunks" in state_update:
                                async for chunk in state_update["final_response_chunks"]:
                                    full_response_text += chunk
                                    yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"

                            if "documents" in state_update:
                                final_references = state_update["documents"]
                    
                    # After the content stream, send the references and the done signal
                    yield f"data: {json.dumps({'type': 'references', 'data': final_references})}\n\n"
                    
                    yield f"data: {json.dumps({'type': 'role', 'role': 'assistant'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            
# In your api/v1/endpoints/chat.py file

        else:
            print("non streaming llll")
            # Convert the simple list of dicts from the frontend into LangGraph's message format
            history = []
            for msg in chat_request.chat_history:
                if msg.get("role") == "user":
                    history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    history.append(AIMessage(content=msg.get("content", "")))

            # Prepare the initial state for the LangGraph agent
            initial_state = {
                "messages": history,
                "userInput": chat_request.message,
                "stream": chat_request.stream
            }
            
            # Invoke the graph and wait for the final state
            final_state = await lg_chat_service.graph.ainvoke(initial_state)

            # (Optional) You can keep the print statement for debugging
            print("\n--- FINAL GRAPH STATE ---")
            import pprint
            pprint.pprint(final_state)
            print("-------------------------\n")

            # --- THIS IS THE CORRECTED LOGIC ---
            # Your node returns a key called "final_response". We get the value from there.
            # print(full_response_text.get("final_content"))
            full_response_text = final_state.get("final_response")
            print("full responseff",full_response_text)

            # Extract references (this part was already working)
            final_references = final_state.get("documents", [])

            # Construct the final response payload
            response_data = {
                "type": "final_response",
                "data": {
                    "content": full_response_text,
                    "references": final_references,
                    "role": "assistant"
                }
            }
            
            # Return a standard JSONResponse
            return JSONResponse(content=response_data)
    except Exception as e:
        print(f"Error in /chat-widget endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
   



@router.post("/chat-lg")
async def send_chat_message_lg(
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Handles sending a message using the LangGraph engine.
    - If `stream: true`, it returns a server-sent event stream.
    - If `stream: false`, it returns a single JSON response.
    """
    try:
        # --- Common Setup (for both stream and non-stream) ---
        chat_session = get_chat_session_for_user(db, chat_request.chat_id, current_user)
        if chat_session.title == "New Chat":
            chat_session.title = chat_request.message[:50]
            db.commit()
        
        add_message_to_db(db, chat_session.id, "user", chat_request.message)

        history = get_chat_history_for_session(db, chat_session.id)
        
        # <<< MINIMAL CHANGE 1: Add 'stream' to the initial state >>>
        initial_state = {
            "messages": history,
            "userInput": chat_request.message,
            "stream": chat_request.stream, # This tells the graph how to behave
        }

        # --- Conditional Logic for Streaming vs. Non-Streaming ---
        if chat_request.stream:
            # <<< This is your existing streaming code, unmodified >>>
            async def generate_stream():
                full_response_text = ""
                final_references = []
                
                print("\n--- Invoking LangGraph Stream ---\n")
                async for event in lg_chat_service.graph.astream(initial_state):
                    for node_name, state_update in event.items():
                        if "final_response_chunks" in state_update:
                            async for chunk in state_update["final_response_chunks"]:
                                full_response_text += chunk
                                yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"

                        if "documents" in state_update and state_update["documents"]:
                            final_references = state_update["documents"]
                
                print("\n" + "="*25 + " Stream Generation Complete " + "="*25)
                
                if full_response_text:
                    assistant_message = add_message_to_db(db, chat_session.id, "assistant", full_response_text)
                    if final_references:
                        add_sources_to_message_in_db(db, assistant_message.id, final_references)

                yield f"data: {json.dumps({'type': 'references', 'data': final_references})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            headers = {"X-Chat-Id": str(chat_session.id), "Access-Control-Expose-Headers": "X-Chat-Id"}
            return StreamingResponse(generate_stream(), media_type="text/event-stream", headers=headers)

        else:
            # <<< MINIMAL CHANGE 2: Add this 'else' block for non-streaming >>>
            print("\n--- Invoking LangGraph Non-Stream (ainvoke) ---\n")
            
            # Use .ainvoke() to run the graph to completion
            final_state = await lg_chat_service.graph.ainvoke(initial_state)

            # Your node returns a single-item generator. We consume it here.
            full_response_text = ""
            if "final_response_chunks" in final_state:
                async for chunk in final_state["final_response_chunks"]:
                    full_response_text += chunk
            
            final_references = final_state.get("documents", [])
            print("\n" + "="*25 + " Non-Stream Invocation Complete " + "="*25)
            
            if full_response_text:
                assistant_message = add_message_to_db(db, chat_session.id, "assistant", full_response_text)
                if final_references:
                    add_sources_to_message_in_db(db, assistant_message.id, final_references)
            
            # Return a standard JSON response
            return JSONResponse(content={
                "message": full_response_text,
                "references": final_references
            })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in /chat-lg endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

        

# @router.post("/chat-lg")
# async def send_chat_message_lg(
#     chat_request: ChatRequest,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Handles sending a message using the new LangGraph engine, with robust stream processing.
#     """
#     if not chat_request.stream:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="The LangGraph endpoint only supports streaming. Please set 'stream: true'."
#         )
#     try:
#         chat_session = get_chat_session_for_user(db, chat_request.chat_id, current_user)
#         if chat_session.title == "New Chat":
#             chat_session.title = chat_request.message[:50]
#             db.commit()
        
#         add_message_to_db(db, chat_session.id, "user", chat_request.message)

#         async def generate_stream():
#             full_response_text = ""
#             final_references = []
#             history = get_chat_history_for_session(db, chat_session.id)
#             initial_state = {
#                 "messages": history,
#                 "userInput": chat_request.message,
#             }

#             pp = pprint.PrettyPrinter(indent=2)
#             print("\n--- Invoking LangGraph Stream ---\n")
            
#             # --- CORRECTED STREAM PROCESSING LOGIC ---
#             async for event in lg_chat_service.graph.astream(initial_state):
#                 print("\n" + "-"*30 + " Graph Event Received " + "-"*30)
                
#                 # The event dictionary's keys are the node names that just ran
#                 for node_name, state_update in event.items():
#                     print(f"NODE EXECUTED: {node_name}")
#                     print("--- STATE UPDATE RETURNED ---")

#                     # Safely check if the update is not None before processing
#                     if state_update is None:
#                         print("Node returned None.")
#                         continue # Skip to the next item in the event

#                     # Print the update for debugging
#                     if 'final_response_chunks' in state_update:
#                         print("{'final_response_chunks': <AsyncGenerator>}")
#                     else:
#                         # pp.pprint(state_update)
#                         print("s")

#                     # --- Process the actual data from the state_update ---
#                     if "final_response_chunks" in state_update:
#                         async for chunk in state_update["final_response_chunks"]:
#                             full_response_text += chunk
#                             yield f"data: {json.dumps({'type': 'chunk', 'data': chunk})}\n\n"

#                     if "documents" in state_update:
#                         final_references = state_update["documents"]
                
#                 print("-"*80)
            
#             print("\n" + "="*25 + " Stream Generation Complete " + "="*25)
#             print(f"Final accumulated response text length: {len(full_response_text)}")
#             print(f"Final number of references found: {len(final_references)}")
#             print("="*75 + "\n")
            
#             if full_response_text:
#                 assistant_message = add_message_to_db(db, chat_session.id, "assistant", full_response_text)
#                 if final_references:
#                     add_sources_to_message_in_db(db, assistant_message.id, final_references)

#             yield f"data: {json.dumps({'type': 'references', 'data': final_references})}\n\n"
#             yield f"data: {json.dumps({'type': 'done'})}\n\n"
#             # print(full_response_text)
#         headers = {"X-Chat-Id": str(chat_session.id), "Access-Control-Expose-Headers": "X-Chat-Id"}
#         return StreamingResponse(generate_stream(), media_type="text/event-stream", headers=headers)
        
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         print(f"Error in /chat-lg endpoint: {e}")
#         raise HTTPException(status_code=500, detail="An internal error occurred.")

@router.post("/chat-delete/{chat_id}")
async def delete_chat_session(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return chat_healpers.delete_chat_session(db, chat_id, current_user.id)




@router.post("/chat-list", response_model=List[ChatSessionInfo])
async def get_user_chat_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1)
):
    """
    Retrieves a list of all chat sessions for the authenticated user.
    """
    return chat_healpers.get_chat_sessions_for_user(
        db, user_id=current_user.id, page=page, size=size
    )

@router.post("/chat-content/{chat_id}", response_model=List[ChatMessageResponse])
async def get_chat_session_history(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the full message history for a specific chat session.
    """
    return chat_healpers.get_messages_for_chat_session(db, chat_id=chat_id, user_id=current_user.id)



@router.get("/chat/stream")
async def chat_with_rag_streaming_get(request: Request):
    """
    Chat with RAG - Streaming response via GET (for simple testing) with automatic index classification
    """
    try:
        print("asdsada")
        body = await request.json()
        message = body.get("message", "")
        hierarchy_filters = body.get("hierarchy_filters", [])
        index_name = body.get("index_name", None)  # Optional - will be auto-classified if not provided
        limit = body.get("limit", 3)
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        
        # Parse hierarchy filters from comma-separated string
        filters = [f.strip() for f in hierarchy_filters.split(",") if f.strip()] if hierarchy_filters else []
        
        chat_service = ChatService()
        
        async def generate_stream():
            async for chunk in chat_service.chat_with_rag_streaming(
                message=message,
                hierarchy_filters=filters,
                index_name=index_name,  # Will be auto-classified if None
                limit=limit
            ):
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
            
            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
