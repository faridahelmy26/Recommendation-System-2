from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, chat, stats
from app.models.database import Database

app = FastAPI(
    title="Ma'man AI Chatbot",
    version="1.0",
    description="AI Chatbot for Ma'man Platform"
)

# ==========================
# CORS Middleware
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Include Routers
# ==========================

app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(stats.router)

# ==========================
# Database
# ==========================

db = Database()
_db_initialized = False

@app.on_event("startup")
def startup_event():
    """Create tables on startup"""
    global _db_initialized
    try:
        db.create_tables()
        _db_initialized = True
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")
        _db_initialized = False

@app.get("/")
def home():
    """Root endpoint"""
    return {
        "message": "Ma'man Chatbot API Running",
        "version": "1.0",
        "status": "active"
    }

@app.get("/health")
def health_check():
    """Health check endpoint - always returns success"""
    return {
        "status": "healthy",
        "database": "connected" if _db_initialized else "initializing",
        "message": "Service is starting up"
    }
