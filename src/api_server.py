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

from .config import CORS_ALLOWED_ORIGINS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline will be initialized at startup
pipeline = None


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.
    Handles generators, pandas Series, numpy types, and NaN values.
    """
    import math
    import numpy as np
    
    # Handle None
    if obj is None:
        return None
    
    # Handle NaN/Infinity FIRST (before other float checks)
    # This catches both Python float and numpy float types
    try:
        if isinstance(obj, (float, np.floating)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
    except (TypeError, ValueError):
        pass
    
    # Handle numpy integer types
    if isinstance(obj, np.integer):
        return int(obj)
    
    # Handle basic JSON types
    if isinstance(obj, (str, int, bool)):
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
            return make_json_serializable(obj.to_dict())
        except Exception:
            pass
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    
    # Handle generators and other iterables (but not strings/bytes)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
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
    evidence_hierarchy: dict = {}
    full_text_articles: list = []  # Articles that received full text (not just abstracts)
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
    allow_origins=CORS_ALLOWED_ORIGINS,
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
        
        # Sanitize result to handle NaN values from pandas/numpy
        return make_json_serializable(result)
        
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
        """Generate SSE events from streaming pipeline in real-time."""
        import threading
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def producer():
            """Runs the synchronous pipeline.answer_streaming in a background thread."""
            try:
                for event in pipeline.answer_streaming(request.query):
                    # Put event into the queue
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
                # Signal completion
                asyncio.run_coroutine_threadsafe(queue.put("[DONE]"), loop)
            except Exception as e:
                logger.error(f"Producer error: {e}", exc_info=True)
                asyncio.run_coroutine_threadsafe(queue.put({"step": "error", "status": "error", "message": str(e)}), loop)
                asyncio.run_coroutine_threadsafe(queue.put("[DONE]"), loop)

        # Start producer thread
        thread = threading.Thread(target=producer)
        thread.start()

        try:
            while True:
                event = await queue.get()
                if event == "[DONE]":
                    break
                
                # Ensure event is JSON-serializable
                serializable_event = make_json_serializable(event)
                
                # Log critical events
                if serializable_event.get("step") == "complete":
                    ans = serializable_event.get("answer", "")
                    logger.info(f"Streaming complete event: status={serializable_event.get('status')}, answer_length={len(ans) if ans else 0}")
                
                # [NEW] Log when sources are sent (helps verify fix)
                if "sources" in serializable_event:
                    logger.info(f"📡 Sending streaming event with {len(serializable_event['sources'])} sources (step: {serializable_event.get('step')})")
                
                data = json.dumps(serializable_event)
                yield f"data: {data}\n\n"
            
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"

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
