from typing import List, Optional, Any, Dict, Union
from qdrant_client import QdrantClient, models

from backend.app.schema.search import SearchFilters
from backend.app.utils.global_state import GlobalState


class QdrantRepository:
    def __init__(self):
        # 依赖注入或单例获取
        self.client = GlobalState.get_db()

    def _build_filters(self, filters: Optional[SearchFilters]) -> Optional[models.Filter]:
        """将前端的 Filter 对象转换为 Qdrant 的 Filter 对象"""
        if not filters:
            return None

        conditions = []
        if filters.year_start is not None:
            conditions.append(models.FieldCondition(key="year", range=models.Range(gte=filters.year_start)))
        if filters.year_end is not None:
            conditions.append(models.FieldCondition(key="year", range=models.Range(lte=filters.year_end)))
        if filters.map_source:
            conditions.append(
                models.FieldCondition(key="source_image", match=models.MatchValue(value=filters.map_source)))
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
               vector_name: str = "",  # 如果使用 named vector
               include_fields: Optional[List[str]] = None,
               exclude_fields: Optional[List[str]] = None,
               hnsw_ef: int = 32
               ) -> List[Any]:
        """
        通用的搜索方法
        """
        # 1. 构建过滤器
        q_filter = self._build_filters(filters)

        # 2. 构建 Payload Selector (性能优化关键)
        payload_selector = None
        if include_fields:
            payload_selector = models.PayloadSelectorInclude(include=include_fields)
        elif exclude_fields:
            payload_selector = models.PayloadSelectorExclude(exclude=exclude_fields)

        # 3. 构建搜索参数
        search_params = models.SearchParams(hnsw_ef=hnsw_ef, exact=False)

        try:
            # 4. 执行查询
            kwargs = {
                "collection_name": collection_name,
                "query": query_vector,
                "query_filter": q_filter,
                "limit": limit,
                "with_payload": payload_selector if payload_selector else True,
                "score_threshold": score_threshold,
                "search_params": search_params
            }

            # 如果指定了 vector_name (针对文档的多向量场景)
            if vector_name:
                kwargs["using"] = vector_name

            hits = self.client.query_points(**kwargs)

            # 兼容性处理：Qdrant 版本差异可能导致返回 tuple 或对象
            if isinstance(hits, tuple): hits = hits[0]
            if hasattr(hits, 'points'): hits = hits.points

            return hits
        except Exception as e:
            # 这里可以记录日志，但最好抛出异常让 Service 层决定如何处理（是重试还是降级）
            print(f"⚠️ [Repo] Qdrant Error in {collection_name}: {e}")
            return []
