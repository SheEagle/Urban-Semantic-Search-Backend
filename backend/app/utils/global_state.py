# app/core_logic/global_state.py
import logging
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings
from backend.app.utils.feature_extractor import PEFeatureExtractor

# Configure logger
logger = logging.getLogger(__name__)


class GlobalState:
    _db_client: QdrantClient = None
    _feature_extractor: PEFeatureExtractor = None
    _text_model = None

    @classmethod
    def get_text_model(cls):
        """
        Singleton for the Sentence Transformer model (MiniLM).
        """
        if cls._text_model is None:
            logger.info("Loading MiniLM for Text-to-Text Search...")
            # Ensure the server has internet access or specify a local model path
            cls._text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return cls._text_model

    @classmethod
    def get_db(cls) -> QdrantClient:
        """
        Qdrant Singleton instance.
        Reuses the connection logic from store_data.py to ensure consistency.
        """
        if cls._db_client is None:
            host = settings.QDRANT_HOST
            port = settings.QDRANT_PORT
            api_key = settings.QDRANT_API_KEY

            logger.info("[Singleton] Connecting to Qdrant...")

            if host.startswith(".") or "/" in host or "\\" in host:
                # Local path mode (Embedded Qdrant)
                cls._db_client = QdrantClient(path=host)
            else:
                # Server mode
                cls._db_client = QdrantClient(
                    host="127.0.0.1",
                    port=port,
                    api_key=api_key,
                    grpc_port=6334,  # Explicitly specify gRPC port
                    prefer_grpc=True  # Force gRPC mode for performance
                )

        return cls._db_client

    @classmethod
    def get_pe_model(cls) -> PEFeatureExtractor:
        """
        Feature Extractor Singleton instance (Positional Encoding).
        """
        if cls._feature_extractor is None:
            logger.info("[Singleton] Initializing Feature Extractor...")
            cls._feature_extractor = PEFeatureExtractor(
                model_name=settings.MODEL_NAME,
                device=settings.DEVICE
            )
            logger.info("[Singleton] Model Ready.")
        return cls._feature_extractor


def init_resources():
    """
    Warm-up function to initialize all singletons during application startup.
    """
    GlobalState.get_db()
    GlobalState.get_pe_model()
    GlobalState.get_text_model()
