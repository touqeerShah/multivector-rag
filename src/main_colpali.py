from fastapi import FastAPI

from src.api.text_routes import build_text_router
from src.api.visual_routes import router as visual_router
from src.core.config import settings


app = FastAPI(title=f"{settings.app_name} (ColPali)")
app.include_router(build_text_router(include_official_colbert=False))
app.include_router(visual_router)
