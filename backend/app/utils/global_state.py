# app/core_logic/global_state.py
from qdrant_client import QdrantClient

from backend.app.core.config import settings
from backend.app.utils.feature_extractor import PEFeatureExtractor


class GlobalState:
    _db_client: QdrantClient = None
    _feature_extractor: PEFeatureExtractor = None

    @classmethod
    def get_db(cls) -> QdrantClient:
        """
        Qdrant 单例
        复用 store_data.py 中的连接判断逻辑
        """
        if cls._db_client is None:
            host = settings.QDRANT_HOST
            port = settings.QDRANT_PORT
            api_key = settings.QDRANT_API_KEY

            print(f"🔌 [Singleton] Connecting to Qdrant...")

            # --- 你的原始 store_data.py 逻辑 ---
            if host.startswith(".") or "/" in host or "\\" in host:
                # 本地路径模式
                cls._db_client = QdrantClient(path=host)
            else:
                # 服务器模式 (Docker)
                cls._db_client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key
                )
            # ----------------------------------

        return cls._db_client

    @classmethod
    def get_model(cls) -> PEFeatureExtractor:
        """
        模型单例
        """
        if cls._feature_extractor is None:
            print(f"⏳ [Singleton] Initializing Feature Extractor...")
            cls._feature_extractor = PEFeatureExtractor(
                model_name=settings.MODEL_NAME,
                device=settings.DEVICE
            )
            print("✅ [Singleton] Model Ready.")
        return cls._feature_extractor


# 初始化函数
def init_resources():
    GlobalState.get_db()
    GlobalState.get_model()
