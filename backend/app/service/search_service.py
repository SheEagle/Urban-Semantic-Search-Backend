# app/services/search_service.py
import io
from typing import Optional

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.schema.search import SearchResultItem, SearchFilters
from backend.app.utils.global_state import GlobalState

from qdrant_client import models


class SearchService:
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME

    def _build_qdrant_filters(self, filters: SearchFilters) -> Optional[models.Filter]:
        """
        辅助函数：将 Pydantic 过滤器转换为 Qdrant Filter 对象
        """
        if not filters:
            return None

        conditions = []

        # 1. 年份范围过滤 (Payload 中必须有 'year' 字段)
        if filters.year_start is not None:
            conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=filters.year_start)
                )
            )
        if filters.year_end is not None:
            conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(lte=filters.year_end)
                )
            )

        # 2. 特定地图来源过滤 (Payload 中必须有 'source_image' 字段)
        if filters.map_source:
            conditions.append(
                models.FieldCondition(
                    key="source_image",
                    match=models.MatchValue(value=filters.map_source)
                )
            )

        # 3. 坐标范围过滤 (Payload 中必须有 'location' Geo 字段)
        # 假设 bbox 格式为 [min_lon, min_lat, max_lon, max_lat]
        if filters.geo_bbox and len(filters.geo_bbox) == 4:
            conditions.append(
                models.FieldCondition(
                    key="location",  # Qdrant 中的 Payload 字段名
                    geo_bounding_box=models.GeoBoundingBox(
                        bottom_right=models.GeoPoint(
                            lon=filters.geo_bbox[2],
                            lat=filters.geo_bbox[1]
                        ),
                        top_left=models.GeoPoint(
                            lon=filters.geo_bbox[0],
                            lat=filters.geo_bbox[3]
                        )
                    )
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

        # 修改通用的执行方法，接收 query_filter

    def _execute_qdrant_search(self, vector_list: list, limit: int, threshold: float,
                               query_filter: models.Filter = None):
        client = GlobalState.get_db()

        hits = client.query_points(
            collection_name=self.collection_name,
            query=vector_list,
            limit=limit,
            query_filter=query_filter
        )

        if hasattr(hits, 'points'):
            hits = hits.points
        elif isinstance(hits, tuple) and hits[0] == 'points':
            hits = hits[1]

        # ... (后续处理 hits 的代码保持不变) ...
        # (略去重复代码，记得返回 results)
        return self._process_hits(hits, threshold)

    def _process_hits(self, hits: list, threshold: float):
        results = []
        # 6. 结果封装 (逻辑同 search_text，可以抽取成一个私有方法 _hits_to_results)
        for i, hit in enumerate(hits):
            if isinstance(hit, tuple): continue
            if not hasattr(hit, 'score'): continue
            if hit.score < threshold: continue

            payload = hit.payload or {}
            loc = payload.get('location', {})

            item = SearchResultItem(
                id=str(hit.id),
                score=hit.score,
                lat=loc.get('lat', 0.0),
                lng=loc.get('lon', 0.0),
                pixel_coords=payload.get('pixel_coords', [0, 0]),
                image_source=payload.get('source_image'),
                geo_polygon=payload.get('geo_detail')
            )
            results.append(item)

        return results

    def search_image(self, image_data: bytes, limit: int, threshold: float) -> list[SearchResultItem]:
        # 1. 获取单例
        client = GlobalState.get_db()
        model = GlobalState.get_model()

        # 2. 图片预处理
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        # 3. 提取特征
        # model.extract_image_features 返回的是 shape 为 (1, 维度) 的 numpy 数组
        feature_array = model.extract_image_features([image])

        # 4. 转换格式
        # 取出第0个元素（因为我们只传了1张图），并转为 python list
        vector_list = feature_array[0].tolist()

        # 5. Qdrant 搜索 (逻辑完全复用 text search，因为都是向量搜向量)
        print(f"🖼️ [Service] Searching Image in '{self.collection_name}'...")
        hits = client.query_points(
            collection_name=self.collection_name,
            query=vector_list,
            limit=limit
        )

        if hasattr(hits, 'points'):
            hits = hits.points
        elif isinstance(hits, tuple) and hits[0] == 'points':
            hits = hits[1]

        results = []

        # 6. 结果封装 (逻辑同 search_text，可以抽取成一个私有方法 _hits_to_results)
        for i, hit in enumerate(hits):
            if isinstance(hit, tuple): continue
            if not hasattr(hit, 'score'): continue
            if hit.score < threshold: continue

            payload = hit.payload or {}
            loc = payload.get('location', {})

            item = SearchResultItem(
                id=str(hit.id),
                score=hit.score,
                lat=loc.get('lat', 0.0),
                lng=loc.get('lon', 0.0),
                pixel_coords=payload.get('pixel_coords', [0, 0]),
                image_source=payload.get('source_image'),
                geo_polygon=payload.get('geo_detail')
            )
            results.append(item)

        return results

    # def search_text(self, query: str, limit: int, threshold: float) -> list[SearchResultItem]:
    #     # 1. 获取单例
    #     client = GlobalState.get_db()
    #     model = GlobalState.get_model()
    #
    #     # 2. 文本编码 (调用 utils)
    #     # 注意：这里会返回 (1, dim) 的 numpy array
    #     raw_vector = model.extract_text_features(query)
    #
    #     # 3. 格式转换 (Numpy -> List)
    #     if hasattr(raw_vector, 'flatten'):
    #         vector_list = raw_vector.flatten().tolist()
    #     elif isinstance(raw_vector, list):
    #         vector_list = raw_vector
    #     else:
    #         vector_list = raw_vector.tolist()
    #
    #     # 4. Qdrant 搜索
    #     print(f"🔍 [Service] Searching in '{self.collection_name}'...")
    #     hits = client.query_points(
    #         collection_name=self.collection_name,
    #         query=vector_list,
    #         limit=limit
    #     )
    #
    #     if hasattr(hits, 'points'):
    #         hits = hits.points
    #     elif isinstance(hits, tuple) and hits[0] == 'points':
    #         # 应对极端情况，如果它本身就是个元组
    #         hits = hits[1]
    #
    #         # 调试打印，确保现在 hits 是个列表
    #     # if isinstance(hits, list) and len(hits) > 0:
    #
    #     results = []
    #
    #     # 3. 遍历结果
    #     for i, hit in enumerate(hits):
    #         # 防御性编程：再次检查 hit 是否为 tuple (应对一些奇怪的迭代器行为)
    #         if isinstance(hit, tuple):
    #             # 如果此时 hit 还是元组 ('points', [...])，说明拆箱没拆干净或者结构嵌套了
    #             # 这种情况下通常跳过或者尝试取值，这里我们做个日志
    #             print(f"⚠️ 跳过异常数据结构 (index {i}): {hit}")
    #             continue
    #
    #         # 正常逻辑：hit 应该是 ScoredPoint 对象
    #         if not hasattr(hit, 'score'):
    #             print(f"⚠️ 跳过无效点 (index {i}), 无 score 属性")
    #             continue
    #
    #         if hit.score < threshold:
    #             continue
    #
    #         payload = hit.payload or {}
    #         loc = payload.get('location', {})
    #
    #         item = SearchResultItem(
    #             id=str(hit.id),
    #             score=hit.score,
    #             lat=loc.get('lat', 0.0),
    #             lng=loc.get('lon', 0.0),
    #             pixel_coords=payload.get('pixel_coords', [0, 0]),
    #             image_source=payload.get('source_image'),
    #             geo_polygon=payload.get('geo_detail')
    #         )
    #         results.append(item)
    #
    #     return results

    def search_text(self, query: str, limit: int, threshold: float, filters: SearchFilters = None):
        model = GlobalState.get_model()
        vector_list = model.extract_text_features(query)[0].tolist()

        # 构建过滤器
        q_filter = self._build_qdrant_filters(filters)

        return self._execute_qdrant_search(vector_list, limit, threshold, query_filter=q_filter)


# 导出实例
search_service = SearchService()
