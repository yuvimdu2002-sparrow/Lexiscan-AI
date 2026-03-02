from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Week 1 - OCR Pipeline")

app.include_router(router)