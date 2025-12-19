"""
FastAPI Server for Medical RAG Pipeline.

Provides REST API endpoints for the Next.js frontend:
- POST /api/chat - Full ScholarQA-style response
- POST /api/chat/stream - Server-Sent Events for streaming
- GET /health - Health check

Run with: uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline will be initialized at startup
pipeline = None


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.
    Handles generators, pandas Series, and other non-serializable types.
    """
    # Handle None and basic JSON types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle dict
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # Handle list and tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle pandas Series/DataFrame
    if hasattr(obj, 'to_dict'):
        try:
            return obj.to_dict()
        except Exception:
            pass
    
    # Handle generators and other iterables (but not strings/bytes)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            # Check if it's actually iterable by trying to get an iterator
            iter(obj)
            return [make_json_serializable(item) for item in obj]
        except (TypeError, ValueError):
            pass
    
    # Handle custom objects with __dict__
    if hasattr(obj, '__dict__'):
        try:
            return make_json_serializable(obj.__dict__)
        except Exception:
            pass
    
    # Fallback: convert to string
    try:
        return str(obj)
    except Exception:
        return None


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str
    stream: bool = False


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    query: str
    report_title: str
    answer: str
    sections: list
    sources: list
    retrieval_stats: dict
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("🚀 Starting Medical RAG API Server...")
    
    try:
        from .rag_pipeline import MedicalRAGPipeline
        pipeline = MedicalRAGPipeline()  # No parameters needed - ScholarQA defaults
        logger.info("✅ Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    logger.info("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Elixir AI Medical RAG API",
    description="ScholarQA-enhanced medical research assistant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline_ready": pipeline is not None}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint for medical questions.
    
    Returns full ScholarQA-style response with sections and citations.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Run pipeline in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            pipeline.answer_scholarqa_style,
            request.query
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with Server-Sent Events.

    Streams progress events and tokens progressively for real-time UI updates.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_generator():
        """Generate SSE events from streaming pipeline."""
        try:
            # Stream events directly from the pipeline
            # Use run_in_executor to run the sync generator in a thread
            loop = asyncio.get_event_loop()
            
            def run_streaming():
                events = []
                for event in pipeline.answer_streaming(request.query):
                    events.append(event)
                return events
            
            # Get all events (this runs the sync generator)
            events = await loop.run_in_executor(None, run_streaming)
            
            # Yield events asynchronously
            for event in events:
                # Log complete events to debug
                if event.get("step") == "complete":
                    logger.info(f"Streaming complete event: status={event.get('status')}, has_answer={bool(event.get('answer'))}, answer_length={len(event.get('answer', '')) if isinstance(event.get('answer'), str) else 0}")
                
                # Ensure event is JSON-serializable
                serializable_event = make_json_serializable(event)
                
                # Verify answer is still present after serialization
                if serializable_event.get("step") == "complete" and "answer" in serializable_event:
                    logger.info(f"After serialization: has_answer={bool(serializable_event.get('answer'))}")
                
                data = json.dumps(serializable_event)
                yield f"data: {data}\n\n"
                # Small delay for UI responsiveness
                await asyncio.sleep(0.05)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_event = json.dumps({
                "step": "error",
                "status": "error",
                "message": str(e)
            })
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/api/query/decompose")
async def decompose_query(request: ChatRequest):
    """
    Debug endpoint to test query decomposition.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.preprocess_query(request.query)
        
        return {
            "original_query": result.original_query,
            "rewritten_query": result.rewritten_query,
            "keyword_query": result.keyword_query,
            "search_filters": result.search_filters,
            "decomposed": result.decomposed.model_dump() if result.decomposed else None
        }
        
    except Exception as e:
        logger.error(f"Decomposition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
