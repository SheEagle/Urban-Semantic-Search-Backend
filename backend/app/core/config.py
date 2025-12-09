import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant Config
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION_NAME: str = "venice_historical_map"
    MAP_COLLECTION: str = "venice_historical_map"
    DOC_COLLECTION: str = "venice_historical_text"

    # Model Config
    MODEL_NAME: str = "PE-Core-B16-224"
    DEVICE: str = "cuda"

    # 允许读取 .env 文件
    class Config:
        env_file = ".env"


# 实例化配置对象
settings = Settings()
