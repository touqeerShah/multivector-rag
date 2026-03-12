from fastapi import FastAPI
from src.api.routes import router
from src.core.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(router)
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host=settings.host, port=settings.port)