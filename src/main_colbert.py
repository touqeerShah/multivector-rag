from fastapi import FastAPI

from src.api.text_routes import build_text_router
from src.core.config import settings


app = FastAPI(title=f"{settings.app_name} (ColBERT)")
app.include_router(build_text_router(include_official_colbert=True))
