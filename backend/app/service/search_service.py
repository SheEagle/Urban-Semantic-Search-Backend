import io
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

import numpy as np
from PIL import Image
from qdrant_client import models

from backend.app.core.config import settings
from backend.app.repository.qdrant_repo import QdrantRepository
from backend.app.schema.search import SearchResultItem, SearchFilters, HeatmapPoint
from backend.app.utils.global_state import GlobalState

# Configure Logger
logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        self.MAP_COLLECTION = settings.MAP_COLLECTION
        self.DOC_COLLECTION = settings.DOC_COLLECTION
        self.repo = QdrantRepository()

    # ==========================================================================
    #  Core Algorithms: Normalization & Helper Functions
    # ==========================================================================

    def _normalize_scores(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """
        Z-Score Normalization (Standardization).
        Formula: z = (x - μ) / σ
        Purpose: Maps scores from different models (Text/Image) onto a standard normal distribution
        so they can be comparably merged.
        """
        if not results or len(results) < 2:
            return results

        # 1. Extract scores
        scores = [r.score for r in results]
        mean = np.mean(scores)
        std = np.std(scores)

        # 2. Defensive check: If standard deviation is 0 (all scores identical), skip normalization
        if std == 0:
            return results

        # 3. Apply normalization
        for r in results:
            r.score = (r.score - mean) / std

        return results

    def _hits_to_results(self, hits, result_type: str, default_content: str = "") -> List[SearchResultItem]:
        """
        Converts raw Qdrant hits into a unified SearchResultItem structure.
        """
        results = []
        # Compatibility handling for different Qdrant client versions
        if isinstance(hits, tuple): hits = hits[0]
        if hasattr(hits, 'points'): hits = hits.points
        if not hits: return results

        for hit in hits:
            if isinstance(hit, tuple) or not hasattr(hit, 'score'): continue

            payload = hit.payload or {}
            loc = payload.get('location', {})

            # Content preview logic
            if result_type == "document":
                content_text = payload.get('content', '')
                content_preview = (content_text[:200] + "...") if len(content_text) > 200 else content_text
            else:
                content_preview = f"{default_content} ({payload.get('year', 'Unknown')})"

            item = SearchResultItem(
                id=str(hit.id),
                score=hit.score,
                year=payload.get('year', 0),
                lat=loc.get('lat', 0.0),
                lng=loc.get('lon', 0.0),
                source_dataset=payload.get('source_dataset') or payload.get('source_image') or 'Unknown',
                content=content_preview,
                fullData=payload,
                type=result_type,
                pixel_coords=payload.get('pixel_coords'),
                image_source=payload.get('source_image')
            )
            results.append(item)
        return results

    def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
        SearchResultItem]:
        """
        Business Logic: Hybrid Text Search.
        Retrieves relevant items from both Document (semantic text) and Map (text-to-visual) collections.
        """
        t_start = time.time()

        # --- 1. Threshold Definitions ---
        DOC_MIN_SCORE = 0.50
        MAP_MIN_SCORE = 0.21

        # --- 2. Model Inference (CPU/GPU) ---
        t_encode = time.time()
        text_vec = []
        pe_vec = []

        try:
            text_vec = GlobalState.get_text_model().encode(query).tolist()
        except Exception as e:
            logger.error(f"Text Model Error: {e}")

        try:
            pe_raw = GlobalState.get_pe_model().extract_text_features(query)
            # Handle potential dimension mismatch
            if hasattr(pe_raw, 'tolist'): pe_raw = pe_raw.tolist()
            if isinstance(pe_raw, list) and isinstance(pe_raw[0], list): pe_raw = pe_raw[0]
            pe_vec = pe_raw
        except Exception as e:
            logger.error(f"PE Model Error: {e}")

        logger.info(f"Encoding Time: {time.time() - t_encode:.4f}s")

        # --- 3. Parallel Database Query (IO Bound) ---

        def fetch_docs():
            if not text_vec: return []
            return self.repo.search(
                collection_name=self.DOC_COLLECTION,
                query_vector=text_vec,
                filters=filters,
                limit=limit * 2,
                score_threshold=DOC_MIN_SCORE,
                vector_name="text_vector",
                hnsw_ef=32
            )

        def fetch_maps():
            if not pe_vec: return []
            return self.repo.search(
                collection_name=self.MAP_COLLECTION,
                query_vector=pe_vec,
                filters=filters,
                limit=limit * 2,
                score_threshold=MAP_MIN_SCORE,
                hnsw_ef=32
            )

        t_search = time.time()
        doc_hits, map_hits = [], []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_doc = executor.submit(fetch_docs)
            future_map = executor.submit(fetch_maps)
            doc_hits = future_doc.result()
            map_hits = future_map.result()

        logger.info(f"IO Search Time: {time.time() - t_search:.4f}s")

        # --- 4. Result Transformation & Normalization ---
        doc_results = self._hits_to_results(doc_hits, "document")
        map_results = self._hits_to_results(map_hits, "map_tile", "Map Fragment")

        if doc_results: self._normalize_scores(doc_results)
        if map_results: self._normalize_scores(map_results)

        # --- 5. Merge & Sort ---
        all_results = doc_results + map_results
        # Only keep results above the mean (Z-score > 0)
        final_results = [r for r in all_results if r.score > 0.75]
        final_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Total Search Time: {time.time() - t_start:.4f}s")
        return final_results[:limit]

    def search_image(self, image_data: bytes, limit: int, threshold: float) -> List[SearchResultItem]:
        """
        Hybrid Image Search (Image -> Image & Text).
        Finds visually similar maps and contextually relevant documents.
        """
        t_start = time.time()

        # --- 1. Threshold Definitions ---
        # Image search generally yields higher confidence scores
        MAP_IMG_MIN_SCORE = 0.40
        DOC_IMG_MIN_SCORE = 0.22

        # --- 2. Image Feature Extraction (CPU Bound) ---
        t_encode = time.time()
        try:
            image = Image.open(io.BytesIO(image_data))
            pe_model = GlobalState.get_pe_model()
            # Extract vector and convert to list
            vector_list = pe_model.extract_image_features([image])[0].tolist()
        except Exception as e:
            logger.error(f"Image Encoding Error: {e}")
            raise ValueError(f"Invalid image processing: {e}")

        logger.info(f"Image Encoding Time: {time.time() - t_encode:.4f}s")

        # --- 3. Parallel Database Query (IO Bound) ---
        def fetch_maps():
            # Image-to-Image (Visual Match)
            return self.repo.search(
                collection_name=self.MAP_COLLECTION,
                query_vector=vector_list,
                limit=limit * 2,
                score_threshold=MAP_IMG_MIN_SCORE,
                hnsw_ef=32
            )

        def fetch_docs():
            # Image-to-Text (Visual -> Description)
            # Must use "pe_vector" (visual alignment vector)
            return self.repo.search(
                collection_name=self.DOC_COLLECTION,
                query_vector=vector_list,
                limit=limit * 2,
                score_threshold=DOC_IMG_MIN_SCORE,
                vector_name="pe_vector",
                hnsw_ef=32
            )

        t_search = time.time()
        map_hits, doc_hits = [], []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_map = executor.submit(fetch_maps)
            future_doc = executor.submit(fetch_docs)

            map_hits = future_map.result()
            doc_hits = future_doc.result()

        logger.info(f"IO Search Time: {time.time() - t_search:.4f}s")

        # --- 4. Result Transformation ---
        map_results = self._hits_to_results(map_hits, "map_tile", "Visual Match")
        doc_results = self._hits_to_results(doc_hits, "document")

        # --- 5. Normalization ---
        if map_results: self._normalize_scores(map_results)
        if doc_results: self._normalize_scores(doc_results)

        # --- 6. Merge & Sort ---
        all_results = map_results + doc_results
        # Filter for quality
        final_results = [r for r in all_results if r.score > 0]
        final_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Total Image Search Time: {time.time() - t_start:.4f}s")
        return final_results[:limit]

    def get_heatmap_data(self, query: Optional[str], limit: int = 2000) -> List[HeatmapPoint]:
        """
        Retrieves lightweight point data for the 3D heatmap.
        If 'query' is provided, returns relevance scores.
        If 'query' is None, returns general data density (random sampling).
        """
        client = GlobalState.get_db()
        payload_selector = models.PayloadSelectorInclude(include=["location"])
        points = []

        # Helper function to process hits
        def process_hits(hits, multiplier=1.0):
            result_points = []
            if isinstance(hits, tuple): hits = hits[0]
            if hasattr(hits, 'points'): hits = hits.points

            for h in hits:
                loc = h.payload.get('location')
                if loc:
                    result_points.append(HeatmapPoint(
                        lat=loc['lat'],
                        lng=loc['lon'],
                        score=h.score * multiplier if query else 1.0  # Use 1.0 score if no query (density mode)
                    ))
            return result_points

        # A. Search Mode (with Query)
        if query:
            # 1. Search Documents
            try:
                text_model = GlobalState.get_text_model()
                vec = text_model.encode(query).tolist()
                hits = client.query_points(
                    self.DOC_COLLECTION, query=vec, using="text_vector",
                    limit=limit // 2, with_payload=payload_selector, score_threshold=0.35
                )
                points.extend(process_hits(hits))
            except Exception as e:
                logger.error(f"Heatmap Doc Search Error: {e}")

            # 2. Search Maps
            try:
                pe_model = GlobalState.get_pe_model()
                vec = pe_model.extract_text_features(query)[0].tolist()
                hits = client.query_points(
                    self.MAP_COLLECTION, query=vec,
                    limit=limit // 2, with_payload=payload_selector, score_threshold=0.20
                )
                # Boost map scores slightly for visual emphasis
                points.extend(process_hits(hits, multiplier=1.1))
            except Exception as e:
                logger.error(f"Heatmap Map Search Error: {e}")

        # B. Density Mode (No Query - General Distribution)
        else:
            try:
                # Scroll (scan) through collections to get random points
                # Note: 'scroll' is more efficient than vector search for random retrieval
                for collection in [self.DOC_COLLECTION, self.MAP_COLLECTION]:
                    res = client.scroll(
                        collection_name=collection,
                        limit=limit // 2,
                        with_payload=payload_selector
                    )
                    points.extend(process_hits(res[0], multiplier=1.0))
            except Exception as e:
                logger.error(f"Heatmap Scroll Error: {e}")

        return points


# Export Singleton
search_service = SearchService()
