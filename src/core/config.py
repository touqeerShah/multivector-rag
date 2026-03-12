from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multivector RAG"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    lancedb_uri: str = "data/lancedb"
    text_table: str = "text_chunks"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def raw_path(self) -> Path:
        path = Path(self.raw_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_path(self) -> Path:
        path = Path(self.processed_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def lancedb_path(self) -> Path:
        path = Path(self.lancedb_uri)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()