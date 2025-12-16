import logging
from typing import List, Optional, Any, Union
from qdrant_client import models

from backend.app.schema.search import SearchFilters
from backend.app.utils.global_state import GlobalState

# Configure logger
logger = logging.getLogger(__name__)


class QdrantRepository:
    def __init__(self):
        # Retrieve the database client via GlobalState (Singleton pattern)
        self.client = GlobalState.get_db()

    def _build_filters(self, filters: Optional[SearchFilters]) -> Optional[models.Filter]:
        """
        Converts the frontend SearchFilters object into a Qdrant Filter object.
        """
        if not filters:
            return None

        conditions = []

        # Filter by Year Range
        if filters.year_start is not None:
            conditions.append(models.FieldCondition(
                key="year",
                range=models.Range(gte=filters.year_start)
            ))
        if filters.year_end is not None:
            conditions.append(models.FieldCondition(
                key="year",
                range=models.Range(lte=filters.year_end)
            ))

        # Filter by Source Map
        if filters.map_source:
            conditions.append(models.FieldCondition(
                key="source_image",
                match=models.MatchValue(value=filters.map_source)
            ))

        # Filter by Geographic Bounding Box
        # Expected format: [min_lon, min_lat, max_lon, max_lat]
        if filters.geo_bbox and len(filters.geo_bbox) == 4:
            conditions.append(
                models.FieldCondition(
                    key="location",
                    geo_bounding_box=models.GeoBoundingBox(
                        bottom_right=models.GeoPoint(lon=filters.geo_bbox[2], lat=filters.geo_bbox[1]),
                        top_left=models.GeoPoint(lon=filters.geo_bbox[0], lat=filters.geo_bbox[3])
                    )
                )
            )

        return models.Filter(must=conditions) if conditions else None

    def search(self,
               collection_name: str,
               query_vector: List[float],
               filters: Optional[SearchFilters] = None,
               limit: int = 10,
               score_threshold: float = 0.0,
               vector_name: str = "",  # For named vectors
               include_fields: Optional[List[str]] = None,
               exclude_fields: Optional[List[str]] = None,
               hnsw_ef: int = 32
               ) -> List[models.ScoredPoint]:
        """
        Generic search method for retrieving points from Qdrant.
        """
        # 1. Build Query Filters
        q_filter = self._build_filters(filters)

        # 2. Build Payload Selector (Critical for network performance)
        payload_selector = None
        if include_fields:
            payload_selector = models.PayloadSelectorInclude(include=include_fields)
        elif exclude_fields:
            payload_selector = models.PayloadSelectorExclude(exclude=exclude_fields)

        # 3. Configure Search Parameters
        search_params = models.SearchParams(hnsw_ef=hnsw_ef, exact=False)

        try:
            # 4. Prepare Arguments
            kwargs = {
                "collection_name": collection_name,
                "query": query_vector,
                "query_filter": q_filter,
                "limit": limit,
                "with_payload": payload_selector if payload_selector else True,
                "score_threshold": score_threshold,
                "search_params": search_params
            }

            # Handle named vectors (e.g., for multi-vector document search)
            if vector_name:
                kwargs["using"] = vector_name

            # 5. Execute Query
            response = self.client.query_points(**kwargs)

            # Return the list of points directly
            return response.points

        except Exception as e:
            # Log the error and return an empty list so the service layer can handle it gracefully
            logger.error(f"[Repo] Qdrant Error in collection '{collection_name}': {e}")
            return []
