
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.config import settings
from app.database.connection import close_pool, open_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown resources."""
    open_pool()
    try:
        yield
    finally:
        close_pool()


# Initialize FastAPI application instance
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered knowledge assistant for Helpdesk",
    version="1.0.0",
    lifespan=lifespan,
)

# Register internal RAG chat routes
app.include_router(chat_router)


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Helpdesk RAG Service gytworkz powered by fastapi",
        "status": "running successfully"
    }


# Health-check endpoint
@app.get("/health")
def health_check():
    return {
        "service": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "status": "healthy"
    }
