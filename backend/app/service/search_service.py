# app/services/search_service.py
import io
from typing import Optional, List

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.schema.search import SearchResultItem, SearchFilters
from backend.app.utils.global_state import GlobalState

from qdrant_client import models


class SearchService:
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME
        self.MAP_COLLECTION = settings.MAP_COLLECTION
        self.DOC_COLLECTION = settings.DOC_COLLECTION

    def _normalize_scores(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """
        对搜索结果的分数进行 Z-Score 归一化 (Standardization)。
        将不同分布的分数映射到均值为0、标准差为1的分布上。
        """
        if not results:
            return results

        # 1. 提取所有分数
        scores = [r.score for r in results]

        # 2. 计算统计量
        mean = np.mean(scores)
        std = np.std(scores)

        # 3. 防御性处理：如果标准差为0（例如只有一个结果，或所有分数相同）
        if std == 0:
            # 这种情况下无法进行 Z-Score，可以选择不处理，或者归一化为 0
            # 这里选择保持原样，或者你可以手动设为 1.0 (如果分数都很高)
            return results

        # 4. 执行归一化
        for r in results:
            # 新分数 = (旧分数 - 均值) / 标准差
            # 注意：这样处理后，分数会有正有负
            r.score = (r.score - mean) / std

        return results

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

    # --- 核心修改：文搜文逻辑 ---
    def _search_documents(self, query: str, limit: int, threshold: float, q_filter: models.Filter) -> List[
        SearchResultItem]:
        """
        使用 MiniLM 模型搜索 venice_docs 集合 (文搜文)
        """
        client = GlobalState.get_db()
        text_model = GlobalState.get_text_model()  # 🔥 获取 MiniLM

        # 1. 生成语义向量
        vector = text_model.encode(query).tolist()

        # 2. 搜索 venice_docs
        hits = client.query_points(
            collection_name=self.DOC_COLLECTION,
            # 🔥 关键修改：query 只传向量值
            query=vector,

            # 🔥 关键修改：用 using 指定向量名称
            using="text_vector",
            query_filter=q_filter,
            limit=limit,
            with_payload=True
        )

        if hasattr(hits, 'points'):
            hits = hits.points
        elif isinstance(hits, tuple) and hits[0] == 'points':
            hits = hits[1]

        results = []

        for i, hit in enumerate(hits):
            if isinstance(hit, tuple): continue
            if not hasattr(hit, 'score'): continue
            if hit.score < threshold: continue

            payload = hit.payload or {}
            loc = payload.get('location', {})

            results.append(SearchResultItem(
                id=str(hit.id),
                score=hit.score,
                lat=loc.get('lat', 0.0),
                lng=loc.get('lon', 0.0),
                image_source=payload.get('source_dataset', 'Document'),
                content=payload.get('content', '')[:200] + "...",  # 截取摘要
                fullData=payload,
                type="document",  # 🔥 标记为文档
                pixel_coords=None
            ))
        return results

        # --- 核心修改：文搜图逻辑 (原有的逻辑微调) ---

    def _search_maps_by_text(self, query: str, limit: int, threshold: float, q_filter: models.Filter) -> List[
        SearchResultItem]:
        """
        使用 PE 模型搜索 venice_maps 集合 (文搜图)
        """
        client = GlobalState.get_db()
        pe_model = GlobalState.get_pe_model()  # 🔥 获取 PE/CLIP

        # 1. 翻译 (可选，建议加上)
        # try:
        #     query = GoogleTranslator(source='auto', target='en').translate(query)
        # except: pass

        # 2. 生成视觉对齐向量
        # extract_text_features 返回 numpy array
        vector_np = pe_model.extract_text_features(query)
        # 处理可能的维度问题 (1, 512) -> [512]
        if hasattr(vector_np, 'tolist'):
            vector_list = vector_np.tolist()
        else:
            vector_list = vector_np

        if isinstance(vector_list[0], list):
            vector_list = vector_list[0]

        # 3. 搜索 venice_maps
        hits = client.query_points(
            collection_name=self.MAP_COLLECTION,
            query=vector_list,  # 🔥 指定 pe_vector
            query_filter=q_filter,
            limit=limit,
            with_payload=True
        )

        if hasattr(hits, 'points'):
            hits = hits.points
        elif isinstance(hits, tuple) and hits[0] == 'points':
            hits = hits[1]

        results = []

        for i, hit in enumerate(hits):
            if isinstance(hit, tuple): continue
            if not hasattr(hit, 'score'): continue
            if hit.score < threshold: continue

            payload = hit.payload or {}
            loc = payload.get('location', {})

            results.append(SearchResultItem(
                id=str(hit.id),
                score=hit.score,
                lat=loc.get('lat', 0.0),
                lng=loc.get('lon', 0.0),
                pixel_coords=payload.get('pixel_coords'),
                image_source=payload.get('source_image'),
                content=f"Map Fragment ({payload.get('year', '')})",
                fullData=payload,
                type="map_tile"  # 🔥 标记为地图切片
            ))
        return results

    # --- 主入口：文本搜索 ---
    def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
        SearchResultItem]:
        q_filter = self._build_qdrant_filters(filters)

        # 定义两个模型各自的“及格线”
        # 经验值：MiniLM 低于 0.4 通常是不相关的
        DOC_MIN_SCORE = 0.35
        # 经验值：CLIP/PE 低于 0.15 通常是随机噪声
        MAP_MIN_SCORE = 0.18
        Z_SCORE_THRESHOLD = 0  # 剔除低于平均水平半个标准差的结果

        doc_results = []
        map_results = []

        # 1. 搜文档
        try:
            # 先拿回来多一点
            raw_docs = self._search_documents(query, limit * 2, 0, q_filter)
            # 🛡️ 第一道防线：绝对阈值过滤
            doc_results = [r for r in raw_docs if r.score > DOC_MIN_SCORE]
        except Exception as e:
            print(f"⚠️ Doc search failed: {e}")

        # 2. 搜地图
        try:
            raw_maps = self._search_maps_by_text(query, limit * 2, 0, q_filter)
            # 🛡️ 第一道防线：绝对阈值过滤
            map_results = [r for r in raw_maps if r.score > MAP_MIN_SCORE]
        except Exception as e:
            print(f"⚠️ Map search failed: {e}")

        # --- 如果某一方被过滤完了，就只剩另一方，避免了强行拉高 ---

        # 3. Z-Score 归一化 (相对排序)
        if doc_results:
            doc_results = self._normalize_scores(doc_results)

        if map_results:
            map_results = self._normalize_scores(map_results)

        # 4. 合并与排序
        all_results = doc_results + map_results
        final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]

        # --- E. 排序与截断 ---
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:2 * limit]

    # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> \
    #         List[SearchResultItem]:
    #     """
    #     聚合搜索：同时搜文档和地图
    #     """
    #     q_filter = self._build_qdrant_filters(filters)
    #
    #     results = []
    #
    #     # 1. 搜文档 (文搜文)
    #     try:
    #         doc_results = self._search_documents(query, limit, threshold, q_filter)
    #         results.extend(doc_results)
    #     except Exception as e:
    #         print(f"⚠️ Doc search failed: {e}")
    #
    #     # 2. 搜地图 (文搜图)
    #     try:
    #         map_results = self._search_maps_by_text(query, limit, threshold, q_filter)
    #         results.extend(map_results)
    #     except Exception as e:
    #         print(f"⚠️ Map search failed: {e}")
    #
    #     # 3. 统一排序 (按分数从高到低)
    #     results.sort(key=lambda x: x.score, reverse=True)
    #
    #     # 4. 截取 Top K
    #     return results[:2 * limit]

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
        model = GlobalState.get_pe_model()

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

    # def search_text(self, query: str, limit: int, threshold: float, filters: SearchFilters = None):
    #     model = GlobalState.get_model()
    #     vector_list = model.extract_text_features(query)[0].tolist()
    #
    #     # 构建过滤器
    #     q_filter = self._build_qdrant_filters(filters)
    #
    #     return self._execute_qdrant_search(vector_list, limit, threshold, query_filter=q_filter)

    def get_heatmap_points(self, query: str = None, year_start: int = None, year_end: int = None, limit: int = 10000):
        client = GlobalState.get_db()
        model = GlobalState.get_model()

        # 1. 构建过滤器 (时间/地图源等)
        # 复用你之前写好的 _build_qdrant_filters
        filters_obj = SearchFilters(year_start=year_start, year_end=year_end)
        q_filter = self._build_qdrant_filters(filters_obj)

        heatmap_data = []

        # --- 分支 A: 搜索模式 (有关键词) ---
        if query:
            # 1. 文本转向量
            vector = model.extract_text_features(query)[0].tolist()

            # 2. 向量搜索
            hits = client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=q_filter,
                limit=limit,  # 这里 limit 可以开大一点
                with_payload=['location'],  # 🔥 关键：只取 location，不要其他大字段
                with_vectors=False
            )

            for hit in hits:
                loc = hit.payload.get('location', {})
                if 'lat' in loc and 'lon' in loc:
                    heatmap_data.append({
                        "lat": loc['lat'],
                        "lng": loc['lon'],
                        "score": hit.score  # 用相似度作为热力权重
                    })

        # --- 分支 B: 全量/浏览模式 (无关键词) ---
        else:
            # 使用 Scroll 接口遍历数据
            # Qdrant 的 scroll 一次最多返回几千条，如果数据量极大需要循环 scroll
            # 这里演示简单的一次性 scroll
            response = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=q_filter,
                limit=limit,
                with_payload=['location'],  # 🔥 关键：只取 location
                with_vectors=False
            )
            points = response[0]  # response 是 (points, offset)

            for point in points:
                loc = point.payload.get('location', {})
                if 'lat' in loc and 'lon' in loc:
                    heatmap_data.append({
                        "lat": loc['lat'],
                        "lng": loc['lon'],
                        "score": 1.0  # 全量模式下，密度即热度，权重设为 1
                    })

        return heatmap_data


# 导出实例
search_service = SearchService()
