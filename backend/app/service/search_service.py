# app/services/search_service.py

import numpy as np

from backend.app.core.config import settings
from backend.app.schema.search import SearchResultItem
from backend.app.utils.global_state import GlobalState


class SearchService:
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME

    def search_text(self, query: str, limit: int, threshold: float) -> list[SearchResultItem]:
        # 1. 获取单例
        client = GlobalState.get_db()
        model = GlobalState.get_model()

        # 2. 文本编码 (调用 utils)
        # 注意：这里会返回 (1, dim) 的 numpy array
        raw_vector = model.extract_text_features(query)

        # 3. 格式转换 (Numpy -> List)
        if hasattr(raw_vector, 'flatten'):
            vector_list = raw_vector.flatten().tolist()
        elif isinstance(raw_vector, list):
            vector_list = raw_vector
        else:
            vector_list = raw_vector.tolist()

        # 4. Qdrant 搜索
        print(f"🔍 [Service] Searching in '{self.collection_name}'...")
        hits = client.query_points(
            collection_name=self.collection_name,
            query=vector_list,
            limit=limit
        )

        if hasattr(hits, 'points'):
            hits = hits.points
        elif isinstance(hits, tuple) and hits[0] == 'points':
            # 应对极端情况，如果它本身就是个元组
            hits = hits[1]

            # 调试打印，确保现在 hits 是个列表
        print(f"📦 [Debug] hits 类型: {type(hits)}")
        if isinstance(hits, list) and len(hits) > 0:
            print(f"🍎 [Debug] 第一个元素类型: {type(hits[0])}")

        results = []

        # 3. 遍历结果
        for i, hit in enumerate(hits):
            # 防御性编程：再次检查 hit 是否为 tuple (应对一些奇怪的迭代器行为)
            if isinstance(hit, tuple):
                # 如果此时 hit 还是元组 ('points', [...])，说明拆箱没拆干净或者结构嵌套了
                # 这种情况下通常跳过或者尝试取值，这里我们做个日志
                print(f"⚠️ 跳过异常数据结构 (index {i}): {hit}")
                continue

            # 正常逻辑：hit 应该是 ScoredPoint 对象
            if not hasattr(hit, 'score'):
                print(f"⚠️ 跳过无效点 (index {i}), 无 score 属性")
                continue

            if hit.score < threshold:
                continue

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


# 导出实例
search_service = SearchService()
