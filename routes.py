import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from retrieve import ask
import os
import logging
import uvicorn

# ----------------------------------- logging --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------------- FastAPI app --------------------------------------
app = FastAPI(
    title="Hadith RAG API",
    description="API for querying Islamic historical traditions using RAG methodology.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ------------------------- API-KEY setup -------------------------
API_KEY = os.getenv("FASTAPI_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")

# ------------------------- Request and Response Models -------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about Islamic history")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the significance of Imam Ali (AS) in Islamic history?"
            }
        }
class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated answer based on retrieved context") 

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "According to the traditions, Janabe Sulaym ibne Qays narrated that..."
            }
        }

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message detailing the issue")

class HealthResponse(BaseModel):
    status: str
    message: str

# ------------------------- API Endpoints -------------------------
@app.get("/",
         response_model=dict, 
         summary="Welcome Endpoint", 
         description="Welcome message and API information", 
         tags=["General"]
    )
async def read_root():
    return {
        "message": "Welcome to the Islamic History RAG API. Use the /ask endpoint to ask questions.",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ask": "POST /ask - Ask questions about Islamic history",
            "health": "GET /health - Check API health"
            }
        }


@app.post("/ask",
        response_model=AnswerResponse,
        summary="Ask a question",
        description="Submit a question about Islamic history and get an answer based on retrieved hadiths and texts",
        dependencies=[Depends(verify_api_key)],
        tags=["Sulaym ibn Qays Traditions"]
    )
async def ask_question(request: QuestionRequest):
    try:
        logger.info(f"Received question: {request.question[:50]}...")
        answer = ask(request.question)
        logger.info("Answer generated successfully")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing the question.")

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API service is running properly",
    dependencies=[Depends(verify_api_key)]
)
async def health_check():
    logger.info("Health check performed")
    return {"status": "ok", "message": "Service is healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)