# # app/services/search_service.py
# import io
# from typing import Optional, List
#
# import numpy as np
# from PIL import Image
#
# from backend.app.core.config import settings
# from backend.app.schema.search import SearchResultItem, SearchFilters, HeatmapPoint
# from backend.app.utils.global_state import GlobalState
#
# from qdrant_client import models
#
#
# class SearchService:
#     def __init__(self):
#         self.collection_name = settings.COLLECTION_NAME
#         self.MAP_COLLECTION = settings.MAP_COLLECTION
#         self.DOC_COLLECTION = settings.DOC_COLLECTION
#
#     def get_heatmap_data(self, query: str, limit: int = 2000) -> List[HeatmapPoint]:
#         """
#         获取轻量级热力图数据 (只返回 lat, lng, score)
#         """
#         client = GlobalState.get_db()
#         text_model = GlobalState.get_text_model()
#
#         points = []
#
#         # 1. 生成搜索向量
#         # 如果 query 为空，理论上应该用 scroll 获取全量分布，这里假设一定有 query
#         try:
#             vector = text_model.encode(query).tolist()
#         except Exception as e:
#             print(f"Embedding failed: {e}")
#             return []
#
#         # 2. 定义只获取 location 字段的过滤器 (性能关键!)
#         # 这样 Qdrant 不会把几 MB 的 full_metadata 传回来
#         payload_selector = models.PayloadSelectorInclude(
#             include=["location", "geo_detail"]
#         )
#
#         # --- A. 搜文档 (Venice Docs) ---
#         try:
#             # 假设 limit 分一半给文档
#             doc_hits = client.query_points(
#                 collection_name=self.DOC_COLLECTION,
#                 # 🔥 关键修改：query 只传向量值
#                 query=vector,
#
#                 # 🔥 关键修改：用 using 指定向量名称
#                 using="text_vector",
#                 limit=limit // 2,
#                 with_payload=payload_selector,  # 🔥 只取坐标
#                 score_threshold=0.35  # 过滤掉完全不相关的
#             )
#
#             if hasattr(doc_hits, 'points'):
#                 doc_hits = doc_hits.points
#             elif isinstance(doc_hits, tuple) and doc_hits[0] == 'points':
#                 doc_hits = doc_hits[1]
#
#             for i, hit in enumerate(doc_hits):
#                 if isinstance(hit, tuple): continue
#                 if not hasattr(hit, 'score'): continue
#
#                 loc = hit.payload.get('location')
#                 # 防御性检查
#                 if loc and 'lat' in loc and 'lon' in loc:
#                     points.append(HeatmapPoint(
#                         lat=loc['lat'],
#                         lng=loc['lon'],
#                         score=hit.score
#                     ))
#
#         except Exception as e:
#             print(f"Heatmap doc search error: {e}")
#
#         # --- B. 搜地图 (Venice Maps) ---
#         # 注意：地图需要 text-to-image 搜索，这里假设你已经把 map_tiles 存入了 text 向量空间
#         # 或者你用的是 CLIP 的 text encoder 搜 image vector
#         # 这里假设使用统一的 text_model 搜索
#         pe_model = GlobalState.get_pe_model()  # 🔥 获取 PE/CLIP
#
#         vector_np = pe_model.extract_text_features(query)
#         # 处理可能的维度问题 (1, 512) -> [512]
#         if hasattr(vector_np, 'tolist'):
#             vector_list = vector_np.tolist()
#         else:
#             vector_list = vector_np
#
#         if isinstance(vector_list[0], list):
#             vector_list = vector_list[0]
#
#         # 3. 搜索 venice_maps
#
#         try:
#             map_hits = client.query_points(
#                 collection_name=self.MAP_COLLECTION,
#                 query=vector_list,  # 🔥 指定 pe_vector
#                 limit=limit // 2,
#                 with_payload=payload_selector,
#                 score_threshold=0.15
#             )
#
#             if hasattr(map_hits, 'points'):
#                 map_hits = map_hits.points
#             elif isinstance(map_hits, tuple) and map_hits[0] == 'points':
#                 map_hits = map_hits[1]
#
#             for i, hit in enumerate(map_hits):
#                 if isinstance(hit, tuple): continue
#                 if not hasattr(hit, 'score'): continue
#
#                 loc = hit.payload.get('location')
#
#                 # 如果 location 没直接在 payload 里，可能要算一下 (针对 Map Tile)
#                 if not loc:
#                     # 尝试从 geo_detail 算中心点
#                     geo = hit.payload.get('geo_detail', {}).get('wgs84', {})
#                     if 'center' in geo:
#                         loc = {'lat': geo['center'][0], 'lon': geo['center'][1]}
#
#                 if loc and 'lat' in loc and 'lon' in loc:
#                     points.append(HeatmapPoint(
#                         lat=loc['lat'],
#                         lng=loc['lon'],
#                         score=hit.score * 1.2  # 给地图一点加权，因为它们通常少
#                     ))
#
#         except Exception as e:
#             print(f"Heatmap map search error: {e}")
#
#         return points
#
#     def _normalize_scores(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
#         """
#         对搜索结果的分数进行 Z-Score 归一化 (Standardization)。
#         将不同分布的分数映射到均值为0、标准差为1的分布上。
#         """
#         if not results:
#             return results
#
#         # 1. 提取所有分数
#         scores = [r.score for r in results]
#
#         # 2. 计算统计量
#         mean = np.mean(scores)
#         std = np.std(scores)
#
#         # 3. 防御性处理：如果标准差为0（例如只有一个结果，或所有分数相同）
#         if std == 0:
#             # 这种情况下无法进行 Z-Score，可以选择不处理，或者归一化为 0
#             # 这里选择保持原样，或者你可以手动设为 1.0 (如果分数都很高)
#             return results
#
#         # 4. 执行归一化
#         for r in results:
#             # 新分数 = (旧分数 - 均值) / 标准差
#             # 注意：这样处理后，分数会有正有负
#             r.score = (r.score - mean) / std
#
#         return results
#
#     def _build_qdrant_filters(self, filters: SearchFilters) -> Optional[models.Filter]:
#         """
#         辅助函数：将 Pydantic 过滤器转换为 Qdrant Filter 对象
#         """
#         if not filters:
#             return None
#
#         conditions = []
#
#         # 1. 年份范围过滤 (Payload 中必须有 'year' 字段)
#         if filters.year_start is not None:
#             conditions.append(
#                 models.FieldCondition(
#                     key="year",
#                     range=models.Range(gte=filters.year_start)
#                 )
#             )
#         if filters.year_end is not None:
#             conditions.append(
#                 models.FieldCondition(
#                     key="year",
#                     range=models.Range(lte=filters.year_end)
#                 )
#             )
#
#         # 2. 特定地图来源过滤 (Payload 中必须有 'source_image' 字段)
#         if filters.map_source:
#             conditions.append(
#                 models.FieldCondition(
#                     key="source_image",
#                     match=models.MatchValue(value=filters.map_source)
#                 )
#             )
#
#         # 3. 坐标范围过滤 (Payload 中必须有 'location' Geo 字段)
#         # 假设 bbox 格式为 [min_lon, min_lat, max_lon, max_lat]
#         if filters.geo_bbox and len(filters.geo_bbox) == 4:
#             conditions.append(
#                 models.FieldCondition(
#                     key="location",  # Qdrant 中的 Payload 字段名
#                     geo_bounding_box=models.GeoBoundingBox(
#                         bottom_right=models.GeoPoint(
#                             lon=filters.geo_bbox[2],
#                             lat=filters.geo_bbox[1]
#                         ),
#                         top_left=models.GeoPoint(
#                             lon=filters.geo_bbox[0],
#                             lat=filters.geo_bbox[3]
#                         )
#                     )
#                 )
#             )
#
#         if not conditions:
#             return None
#
#         return models.Filter(must=conditions)
#
#         # 修改通用的执行方法，接收 query_filter
#
#     # --- 核心修改：文搜文逻辑 ---
#     def _search_documents(self, query: str, limit: int, threshold: float, q_filter: models.Filter) -> List[
#         SearchResultItem]:
#         """
#         使用 MiniLM 模型搜索 venice_docs 集合 (文搜文)
#         """
#         client = GlobalState.get_db()
#         text_model = GlobalState.get_text_model()  # 🔥 获取 MiniLM
#
#         # 1. 生成语义向量
#         vector = text_model.encode(query).tolist()
#
#         # 2. 搜索 venice_docs
#         hits = client.query_points(
#             collection_name=self.DOC_COLLECTION,
#             # 🔥 关键修改：query 只传向量值
#             query=vector,
#
#             # 🔥 关键修改：用 using 指定向量名称
#             using="text_vector",
#             query_filter=q_filter,
#             limit=limit,
#             with_payload=True
#         )
#
#         if hasattr(hits, 'points'):
#             hits = hits.points
#         elif isinstance(hits, tuple) and hits[0] == 'points':
#             hits = hits[1]
#
#         results = []
#
#         for i, hit in enumerate(hits):
#             if isinstance(hit, tuple): continue
#             if not hasattr(hit, 'score'): continue
#             if hit.score < threshold: continue
#
#             payload = hit.payload or {}
#             loc = payload.get('location', {})
#
#             results.append(SearchResultItem(
#                 id=str(hit.id),
#                 score=hit.score,
#                 year=payload.get('year', 0),
#                 lat=loc.get('lat', 0.0),
#                 lng=loc.get('lon', 0.0),
#                 source_dataset=payload.get('source_dataset', 'Document'),
#                 content=payload.get('content', '')[:200] + "...",  # 截取摘要
#                 fullData=payload,
#                 type="document",  # 🔥 标记为文档
#                 pixel_coords=None
#             ))
#         return results
#
#         # --- 核心修改：文搜图逻辑 (原有的逻辑微调) ---
#
#     def _search_maps_by_text(self, query: str, limit: int, threshold: float, q_filter: models.Filter) -> List[
#         SearchResultItem]:
#         """
#         使用 PE 模型搜索 venice_maps 集合 (文搜图)
#         """
#         client = GlobalState.get_db()
#         pe_model = GlobalState.get_pe_model()  # 🔥 获取 PE/CLIP
#
#         # 1. 翻译 (可选，建议加上)
#         # try:
#         #     query = GoogleTranslator(source='auto', target='en').translate(query)
#         # except: pass
#
#         # 2. 生成视觉对齐向量
#         # extract_text_features 返回 numpy array
#         vector_np = pe_model.extract_text_features(query)
#         # 处理可能的维度问题 (1, 512) -> [512]
#         if hasattr(vector_np, 'tolist'):
#             vector_list = vector_np.tolist()
#         else:
#             vector_list = vector_np
#
#         if isinstance(vector_list[0], list):
#             vector_list = vector_list[0]
#
#         # 3. 搜索 venice_maps
#         hits = client.query_points(
#             collection_name=self.MAP_COLLECTION,
#             query=vector_list,  # 🔥 指定 pe_vector
#             query_filter=q_filter,
#             limit=limit,
#             with_payload=True
#         )
#
#         if hasattr(hits, 'points'):
#             hits = hits.points
#         elif isinstance(hits, tuple) and hits[0] == 'points':
#             hits = hits[1]
#
#         results = []
#
#         for i, hit in enumerate(hits):
#             if isinstance(hit, tuple): continue
#             if not hasattr(hit, 'score'): continue
#             if hit.score < threshold: continue
#
#             payload = hit.payload or {}
#             loc = payload.get('location', {})
#
#             results.append(SearchResultItem(
#                 id=str(hit.id),
#                 score=hit.score,
#                 year=payload.get('year', 0),
#                 lat=loc.get('lat', 0.0),
#                 lng=loc.get('lon', 0.0),
#                 pixel_coords=payload.get('pixel_coords'),
#                 image_source=payload.get('source_image'),
#                 content=f"Map Fragment ({payload.get('year', '')})",
#                 fullData=payload,
#                 type="map_tile"  # 🔥 标记为地图切片
#             ))
#         return results
#
#     # --- 主入口：文本搜索 ---
#     def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
#         SearchResultItem]:
#         q_filter = self._build_qdrant_filters(filters)
#
#         # 定义两个模型各自的“及格线”
#         # 经验值：MiniLM 低于 0.4 通常是不相关的
#         DOC_MIN_SCORE = 0.35
#         # 经验值：CLIP/PE 低于 0.15 通常是随机噪声
#         MAP_MIN_SCORE = 0.18
#         Z_SCORE_THRESHOLD = 0  # 剔除低于平均水平半个标准差的结果
#
#         doc_results = []
#         map_results = []
#
#         # 1. 搜文档
#         try:
#             # 先拿回来多一点
#             raw_docs = self._search_documents(query, limit * 2, 0, q_filter)
#             # 🛡️ 第一道防线：绝对阈值过滤
#             doc_results = [r for r in raw_docs if r.score > DOC_MIN_SCORE]
#         except Exception as e:
#             print(f"⚠️ Doc search failed: {e}")
#
#         # 2. 搜地图
#         try:
#             raw_maps = self._search_maps_by_text(query, limit * 2, 0, q_filter)
#             # 🛡️ 第一道防线：绝对阈值过滤
#             map_results = [r for r in raw_maps if r.score > MAP_MIN_SCORE]
#         except Exception as e:
#             print(f"⚠️ Map search failed: {e}")
#
#         # --- 如果某一方被过滤完了，就只剩另一方，避免了强行拉高 ---
#
#         # 3. Z-Score 归一化 (相对排序)
#         if doc_results:
#             doc_results = self._normalize_scores(doc_results)
#
#         if map_results:
#             map_results = self._normalize_scores(map_results)
#
#         # 4. 合并与排序
#         all_results = doc_results + map_results
#         final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]
#
#         # --- E. 排序与截断 ---
#         final_results.sort(key=lambda x: x.score, reverse=True)
#         return final_results[:2 * limit]
#
#     # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> \
#     #         List[SearchResultItem]:
#     #     """
#     #     聚合搜索：同时搜文档和地图
#     #     """
#     #     q_filter = self._build_qdrant_filters(filters)
#     #
#     #     results = []
#     #
#     #     # 1. 搜文档 (文搜文)
#     #     try:
#     #         doc_results = self._search_documents(query, limit, threshold, q_filter)
#     #         results.extend(doc_results)
#     #     except Exception as e:
#     #         print(f"⚠️ Doc search failed: {e}")
#     #
#     #     # 2. 搜地图 (文搜图)
#     #     try:
#     #         map_results = self._search_maps_by_text(query, limit, threshold, q_filter)
#     #         results.extend(map_results)
#     #     except Exception as e:
#     #         print(f"⚠️ Map search failed: {e}")
#     #
#     #     # 3. 统一排序 (按分数从高到低)
#     #     results.sort(key=lambda x: x.score, reverse=True)
#     #
#     #     # 4. 截取 Top K
#     #     return results[:2 * limit]
#
#     def _execute_qdrant_search(self, vector_list: list, limit: int, threshold: float,
#                                query_filter: models.Filter = None):
#         client = GlobalState.get_db()
#
#         hits = client.query_points(
#             collection_name=self.collection_name,
#             query=vector_list,
#             limit=limit,
#             query_filter=query_filter
#         )
#
#         if hasattr(hits, 'points'):
#             hits = hits.points
#         elif isinstance(hits, tuple) and hits[0] == 'points':
#             hits = hits[1]
#
#         return self._process_hits(hits, threshold)
#
#     def _process_hits(self, hits: list, threshold: float):
#         results = []
#         # 6. 结果封装 (逻辑同 search_text，可以抽取成一个私有方法 _hits_to_results)
#         for i, hit in enumerate(hits):
#             if isinstance(hit, tuple): continue
#             if not hasattr(hit, 'score'): continue
#             if hit.score < threshold: continue
#
#             payload = hit.payload or {}
#             loc = payload.get('location', {})
#
#             item = SearchResultItem(
#                 id=str(hit.id),
#                 score=hit.score,
#                 lat=loc.get('lat', 0.0),
#                 lng=loc.get('lon', 0.0),
#                 pixel_coords=payload.get('pixel_coords', [0, 0]),
#                 image_source=payload.get('source_image'),
#                 geo_polygon=payload.get('geo_detail')
#             )
#             results.append(item)
#
#         return results
#
#     def search_image(self, image_data: bytes, limit: int, threshold: float) -> list[SearchResultItem]:
#         # 1. 获取单例
#         client = GlobalState.get_db()
#         model = GlobalState.get_pe_model()
#
#         # 2. 图片预处理
#         try:
#             image = Image.open(io.BytesIO(image_data))
#         except Exception as e:
#             raise ValueError(f"Invalid image file: {e}")
#
#         # 3. 提取特征
#         # model.extract_image_features 返回的是 shape 为 (1, 维度) 的 numpy 数组
#         feature_array = model.extract_image_features([image])
#
#         # 4. 转换格式
#         # 取出第0个元素（因为我们只传了1张图），并转为 python list
#         vector_list = feature_array[0].tolist()
#
#         # 5. Qdrant 搜索 (逻辑完全复用 text search，因为都是向量搜向量)
#         print(f"🖼️ [Service] Searching Image in '{self.collection_name}'...")
#         hits = client.query_points(
#             collection_name=self.collection_name,
#             query=vector_list,
#             limit=limit
#         )
#
#         if hasattr(hits, 'points'):
#             hits = hits.points
#         elif isinstance(hits, tuple) and hits[0] == 'points':
#             hits = hits[1]
#
#         results = []
#
#         # 6. 结果封装 (逻辑同 search_text，可以抽取成一个私有方法 _hits_to_results)
#         for i, hit in enumerate(hits):
#             if isinstance(hit, tuple): continue
#             if not hasattr(hit, 'score'): continue
#             if hit.score < threshold: continue
#
#             payload = hit.payload or {}
#             loc = payload.get('location', {})
#
#             item = SearchResultItem(
#                 id=str(hit.id),
#                 score=hit.score,
#                 lat=loc.get('lat', 0.0),
#                 lng=loc.get('lon', 0.0),
#                 pixel_coords=payload.get('pixel_coords', [0, 0]),
#                 image_source=payload.get('source_image'),
#                 geo_polygon=payload.get('geo_detail')
#             )
#             results.append(item)
#
#         return results
#
#     # def search_text(self, query: str, limit: int, threshold: float) -> list[SearchResultItem]:
#     #     # 1. 获取单例
#     #     client = GlobalState.get_db()
#     #     model = GlobalState.get_model()
#     #
#     #     # 2. 文本编码 (调用 utils)
#     #     # 注意：这里会返回 (1, dim) 的 numpy array
#     #     raw_vector = model.extract_text_features(query)
#     #
#     #     # 3. 格式转换 (Numpy -> List)
#     #     if hasattr(raw_vector, 'flatten'):
#     #         vector_list = raw_vector.flatten().tolist()
#     #     elif isinstance(raw_vector, list):
#     #         vector_list = raw_vector
#     #     else:
#     #         vector_list = raw_vector.tolist()
#     #
#     #     # 4. Qdrant 搜索
#     #     print(f"🔍 [Service] Searching in '{self.collection_name}'...")
#     #     hits = client.query_points(
#     #         collection_name=self.collection_name,
#     #         query=vector_list,
#     #         limit=limit
#     #     )
#     #
#     #     if hasattr(hits, 'points'):
#     #         hits = hits.points
#     #     elif isinstance(hits, tuple) and hits[0] == 'points':
#     #         # 应对极端情况，如果它本身就是个元组
#     #         hits = hits[1]
#     #
#     #         # 调试打印，确保现在 hits 是个列表
#     #     # if isinstance(hits, list) and len(hits) > 0:
#     #
#     #     results = []
#     #
#     #     # 3. 遍历结果
#     #     for i, hit in enumerate(hits):
#     #         # 防御性编程：再次检查 hit 是否为 tuple (应对一些奇怪的迭代器行为)
#     #         if isinstance(hit, tuple):
#     #             # 如果此时 hit 还是元组 ('points', [...])，说明拆箱没拆干净或者结构嵌套了
#     #             # 这种情况下通常跳过或者尝试取值，这里我们做个日志
#     #             print(f"⚠️ 跳过异常数据结构 (index {i}): {hit}")
#     #             continue
#     #
#     #         # 正常逻辑：hit 应该是 ScoredPoint 对象
#     #         if not hasattr(hit, 'score'):
#     #             print(f"⚠️ 跳过无效点 (index {i}), 无 score 属性")
#     #             continue
#     #
#     #         if hit.score < threshold:
#     #             continue
#     #
#     #         payload = hit.payload or {}
#     #         loc = payload.get('location', {})
#     #
#     #         item = SearchResultItem(
#     #             id=str(hit.id),
#     #             score=hit.score,
#     #             lat=loc.get('lat', 0.0),
#     #             lng=loc.get('lon', 0.0),
#     #             pixel_coords=payload.get('pixel_coords', [0, 0]),
#     #             image_source=payload.get('source_image'),
#     #             geo_polygon=payload.get('geo_detail')
#     #         )
#     #         results.append(item)
#     #
#     #     return results
#
#     # def search_text(self, query: str, limit: int, threshold: float, filters: SearchFilters = None):
#     #     model = GlobalState.get_model()
#     #     vector_list = model.extract_text_features(query)[0].tolist()
#     #
#     #     # 构建过滤器
#     #     q_filter = self._build_qdrant_filters(filters)
#     #
#     #     return self._execute_qdrant_search(vector_list, limit, threshold, query_filter=q_filter)
#
#     def get_heatmap_points(self, query: str = None, year_start: int = None, year_end: int = None, limit: int = 10000):
#         client = GlobalState.get_db()
#         model = GlobalState.get_model()
#
#         # 1. 构建过滤器 (时间/地图源等)
#         # 复用你之前写好的 _build_qdrant_filters
#         filters_obj = SearchFilters(year_start=year_start, year_end=year_end)
#         q_filter = self._build_qdrant_filters(filters_obj)
#
#         heatmap_data = []
#
#         # --- 分支 A: 搜索模式 (有关键词) ---
#         if query:
#             # 1. 文本转向量
#             vector = model.extract_text_features(query)[0].tolist()
#
#             # 2. 向量搜索
#             hits = client.search(
#                 collection_name=self.collection_name,
#                 query_vector=vector,
#                 query_filter=q_filter,
#                 limit=limit,  # 这里 limit 可以开大一点
#                 with_payload=['location'],  # 🔥 关键：只取 location，不要其他大字段
#                 with_vectors=False
#             )
#
#             for hit in hits:
#                 loc = hit.payload.get('location', {})
#                 if 'lat' in loc and 'lon' in loc:
#                     heatmap_data.append({
#                         "lat": loc['lat'],
#                         "lng": loc['lon'],
#                         "score": hit.score  # 用相似度作为热力权重
#                     })
#
#         # --- 分支 B: 全量/浏览模式 (无关键词) ---
#         else:
#             # 使用 Scroll 接口遍历数据
#             # Qdrant 的 scroll 一次最多返回几千条，如果数据量极大需要循环 scroll
#             # 这里演示简单的一次性 scroll
#             response = client.scroll(
#                 collection_name=self.collection_name,
#                 scroll_filter=q_filter,
#                 limit=limit,
#                 with_payload=['location'],  # 🔥 关键：只取 location
#                 with_vectors=False
#             )
#             points = response[0]  # response 是 (points, offset)
#
#             for point in points:
#                 loc = point.payload.get('location', {})
#                 if 'lat' in loc and 'lon' in loc:
#                     heatmap_data.append({
#                         "lat": loc['lat'],
#                         "lng": loc['lon'],
#                         "score": 1.0  # 全量模式下，密度即热度，权重设为 1
#                     })
#
#         return heatmap_data
#
#
# # 导出实例
# search_service = SearchService()


# app/services/search_service.py
# import io
# from typing import Optional, List, Union
#
# import numpy as np
# from PIL import Image
#
# from backend.app.core.config import settings
# from backend.app.schema.search import SearchResultItem, SearchFilters, HeatmapPoint
# from backend.app.utils.global_state import GlobalState
#
# from qdrant_client import models
#
#
# # from qdrant_client.http.models import PointStruct, ScoredPoint # 根据版本可能需要引入
#
#
# class SearchService:
#     def __init__(self):
#         # 从配置中读取集合名称
#         self.MAP_COLLECTION = settings.MAP_COLLECTION  # 地图切片集合 (visual vector)
#         self.DOC_COLLECTION = settings.DOC_COLLECTION  # 历史文献集合 (text vector + pe_vector)
#
#     # ==========================================================================
#     #  核心：辅助方法 (Helpers)
#     # ==========================================================================
#
#     def _normalize_scores(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
#         """
#         Z-Score 归一化：将不同分布的分数映射到统一标准，以便混合排序。
#         """
#         if not results:
#             return results
#
#         scores = [r.score for r in results]
#         mean = np.mean(scores)
#         std = np.std(scores)
#
#         if std == 0:
#             return results
#
#         for r in results:
#             r.score = (r.score - mean) / std
#
#         return results
#
#     def _build_qdrant_filters(self, filters: SearchFilters) -> Optional[models.Filter]:
#         """
#         构建 Qdrant 过滤器 (时间、空间、来源)
#         """
#         if not filters:
#             return None
#
#         conditions = []
#
#         # 1. 年份范围
#         if filters.year_start is not None:
#             conditions.append(models.FieldCondition(key="year", range=models.Range(gte=filters.year_start)))
#         if filters.year_end is not None:
#             conditions.append(models.FieldCondition(key="year", range=models.Range(lte=filters.year_end)))
#
#         # 2. 地图来源
#         if filters.map_source:
#             conditions.append(
#                 models.FieldCondition(key="source_image", match=models.MatchValue(value=filters.map_source)))
#
#         # 3. 空间范围 (BBox)
#         if filters.geo_bbox and len(filters.geo_bbox) == 4:
#             conditions.append(
#                 models.FieldCondition(
#                     key="location",
#                     geo_bounding_box=models.GeoBoundingBox(
#                         bottom_right=models.GeoPoint(lon=filters.geo_bbox[2], lat=filters.geo_bbox[1]),
#                         top_left=models.GeoPoint(lon=filters.geo_bbox[0], lat=filters.geo_bbox[3])
#                     )
#                 )
#             )
#
#         return models.Filter(must=conditions) if conditions else None
#
#     def _hits_to_results(self, hits, threshold: float, result_type: str, default_content: str = "") -> List[
#         SearchResultItem]:
#         """
#         通用结果转换：将 Qdrant Point 转换为 SearchResultItem
#         """
#         results = []
#
#         # 兼容性处理：如果返回的是 tuple 结构 (points, offset)
#         if isinstance(hits, tuple):
#             hits = hits[0]
#         # 兼容性处理：如果返回的是对象且包含 points 属性
#         if hasattr(hits, 'points'):
#             hits = hits.points
#
#         if not hits:
#             return results
#
#         for hit in hits:
#             # 防御性跳过
#             if isinstance(hit, tuple) or not hasattr(hit, 'score'):
#                 continue
#
#             if hit.score < threshold:
#                 continue
#
#             payload = hit.payload or {}
#             loc = payload.get('location', {})
#
#             # 区分内容展示
#             if result_type == "document":
#                 # 文档显示 content 字段摘要
#                 content_preview = payload.get('content', '')[:200] + "..."
#             else:
#                 # 地图显示预设标题
#                 content_preview = f"{default_content} ({payload.get('year', 'Unknown')})"
#
#             item = SearchResultItem(
#                 id=str(hit.id),
#                 score=hit.score,
#                 year=payload.get('year', 0),
#                 lat=loc.get('lat', 0.0),
#                 lng=loc.get('lon', 0.0),
#                 source_dataset=payload.get('source_dataset') or payload.get('source_image') or 'Unknown',
#                 content=content_preview,
#                 fullData=payload,
#                 type=result_type,
#                 pixel_coords=payload.get('pixel_coords'),  # 地图特有
#                 geo_polygon=payload.get('geo_detail')  # 地图特有
#             )
#             results.append(item)
#
#         return results

# ==========================================================================
#  功能 1: 文本搜索 (Text Search) - 混合搜索
# ==========================================================================

# def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
#     SearchResultItem]:
#     """
#     文本 -> (文搜文 MiniLM) + (文搜图 PE/CLIP)
#     """
#     client = GlobalState.get_db()
#     q_filter = self._build_qdrant_filters(filters)
#
#     # 阈值设定
#     DOC_MIN_SCORE = 0.35  # 文档相关性阈值
#     MAP_MIN_SCORE = 0.18  # 地图相关性阈值
#
#     doc_results = []
#     map_results = []
#
#     # --- A. 搜文档 (MiniLM) ---
#     try:
#         text_model = GlobalState.get_text_model()
#         text_vec = text_model.encode(query).tolist()
#
#         hits_doc = client.query_points(
#             collection_name=self.DOC_COLLECTION,
#             query=text_vec,
#             using="text_vector",  # 显式指定语义向量
#             query_filter=q_filter,
#             limit=limit * 2,  # 多取一点用于过滤
#             with_payload=True
#         )
#         doc_results = self._hits_to_results(hits_doc, DOC_MIN_SCORE, "document")
#     except Exception as e:
#         print(f"⚠️ Doc search (Text) failed: {e}")
#
#     # --- B. 搜地图 (PE/CLIP) ---
#     try:
#         pe_model = GlobalState.get_pe_model()
#         # 获取文本的视觉特征向量
#         pe_vec = pe_model.extract_text_features(query)
#         # 格式清洗
#         if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
#         if isinstance(pe_vec, list) and len(pe_vec) == 1 and isinstance(pe_vec[0], list):
#             pe_vec = pe_vec[0]
#
#         hits_map = client.query_points(
#             collection_name=self.MAP_COLLECTION,
#             query=pe_vec,
#             # maps 集合通常使用默认向量，如果定义了名字需加 using="pe_vector"
#             query_filter=q_filter,
#             limit=limit * 2,
#             with_payload=True
#         )
#         # 地图结果
#         map_results = self._hits_to_results(hits_map, MAP_MIN_SCORE, "map_tile", "Map Fragment")
#     except Exception as e:
#         print(f"⚠️ Map search (Text) failed: {e}")
#
#     # --- C. 归一化与合并 ---
#     if doc_results: doc_results = self._normalize_scores(doc_results)
#     if map_results: map_results = self._normalize_scores(map_results)
#
#     all_results = doc_results + map_results
#     all_results.sort(key=lambda x: x.score, reverse=True)
#
#     return all_results[:limit]
#
# # ==========================================================================
# #  功能 2: 图片搜索 (Image Search) - 混合搜索 (NEW!)
# # ==========================================================================
#
# def search_image(self, image_data: bytes, limit: int, threshold: float) -> List[SearchResultItem]:
#     """
#     图片 -> (图搜图 Maps) + (图搜文 Docs)
#     """
#     client = GlobalState.get_db()
#     pe_model = GlobalState.get_pe_model()
#
#     # 1. 图片预处理与向量化
#     try:
#         image = Image.open(io.BytesIO(image_data))
#         # extract_image_features 返回 shape (1, dim)
#         feature_array = pe_model.extract_image_features([image])
#
#         # 转换为 list [dim]
#         vector_list = feature_array[0].tolist()
#     except Exception as e:
#         print(f"❌ Image processing failed: {e}")
#         raise ValueError(f"Invalid image: {e}")
#
#     # 阈值设定 (图片搜索通常置信度较高，阈值可以稍高)
#     MAP_IMG_MIN_SCORE = 0.4
#     DOC_IMG_MIN_SCORE = 0.4
#
#     doc_results = []
#     map_results = []
#
#     print(f"🖼️ [Search] Start Image Search...")
#
#     # --- A. 搜地图 (图搜图: Image -> Image Vector) ---
#     try:
#         hits_map = client.query_points(
#             collection_name=self.MAP_COLLECTION,
#             query=vector_list,
#             # map 集合只有视觉向量，通常是默认向量
#             limit=limit * 2,
#             with_payload=True
#         )
#         map_results = self._hits_to_results(hits_map, MAP_IMG_MIN_SCORE, "map_tile", "Visual Match Map")
#     except Exception as e:
#         print(f"⚠️ Map search (Image) failed: {e}")
#
#     # --- B. 搜文档 (图搜文: Image -> Text's PE Vector) ---
#     # 前提：DOC_COLLECTION 在入库时计算并存储了 'pe_vector'
#     try:
#         hits_doc = client.query_points(
#             collection_name=self.DOC_COLLECTION,
#             query=vector_list,
#             using="pe_vector",  # 🔥 关键：指定使用文档的视觉对齐向量
#             limit=limit * 2,
#             with_payload=True
#         )
#         doc_results = self._hits_to_results(hits_doc, DOC_IMG_MIN_SCORE, "document")
#     except Exception as e:
#         # 如果文档集合里没有 pe_vector，这里会报错，捕捉它不影响地图搜索
#         print(f"⚠️ Doc search (Image) failed (Maybe 'pe_vector' missing?): {e}")
#
#     # --- C. 归一化与合并 ---
#     # 即使是图片搜索，不同集合的相似度分布也可能不同，建议归一化
#     if doc_results: doc_results = self._normalize_scores(doc_results)
#     if map_results: map_results = self._normalize_scores(map_results)
#
#     all_results = map_results + doc_results
#     all_results.sort(key=lambda x: x.score, reverse=True)
#
#     return all_results[:limit]
# import io
# from typing import Optional, List
#
# import numpy as np
# from PIL import Image
#
# from backend.app.core.config import settings
# from backend.app.schema.search import SearchResultItem, SearchFilters, HeatmapPoint
# from backend.app.utils.global_state import GlobalState
#
# from qdrant_client import models
#
# class SearchService:
#     def __init__(self):
#         self.MAP_COLLECTION = settings.MAP_COLLECTION
#         self.DOC_COLLECTION = settings.DOC_COLLECTION
#
#     # ==========================================================================
#     #  核心算法: 归一化与辅助函数
#     # ==========================================================================
#
#     def _normalize_scores(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
#         """
#         Z-Score 归一化 (Standardization)
#         公式: z = (x - μ) / σ
#         作用: 将不同模型的分数映射到同一个标准正态分布上，使它们可以相互比较。
#         """
#         if not results or len(results) < 2:
#             return results
#
#         # 1. 提取分数
#         scores = [r.score for r in results]
#         mean = np.mean(scores)
#         std = np.std(scores)
#
#         # 2. 防御性处理：如果标准差为0 (所有分数都一样)，无法归一化
#         if std == 0:
#             return results
#
#         # 3. 执行归一化
#         for r in results:
#             r.score = (r.score - mean) / std
#
#         return results
#
#     def _build_qdrant_filters(self, filters: SearchFilters) -> Optional[models.Filter]:
#         """构建 Qdrant 过滤器"""
#         if not filters:
#             return None
#
#         conditions = []
#         if filters.year_start is not None:
#             conditions.append(models.FieldCondition(key="year", range=models.Range(gte=filters.year_start)))
#         if filters.year_end is not None:
#             conditions.append(models.FieldCondition(key="year", range=models.Range(lte=filters.year_end)))
#         if filters.map_source:
#             conditions.append(
#                 models.FieldCondition(key="source_image", match=models.MatchValue(value=filters.map_source)))
#         if filters.geo_bbox and len(filters.geo_bbox) == 4:
#             conditions.append(
#                 models.FieldCondition(
#                     key="location",
#                     geo_bounding_box=models.GeoBoundingBox(
#                         bottom_right=models.GeoPoint(lon=filters.geo_bbox[2], lat=filters.geo_bbox[1]),
#                         top_left=models.GeoPoint(lon=filters.geo_bbox[0], lat=filters.geo_bbox[3])
#                     )
#                 )
#             )
#         return models.Filter(must=conditions) if conditions else None
#
#     def _hits_to_results(self, hits, result_type: str, default_content: str = "") -> List[SearchResultItem]:
#         """将 Qdrant 返回的原始 hits 转换为统一的数据结构"""
#         results = []
#         if isinstance(hits, tuple): hits = hits[0]
#         if hasattr(hits, 'points'): hits = hits.points
#         if not hits: return results
#
#         for hit in hits:
#             if isinstance(hit, tuple) or not hasattr(hit, 'score'): continue
#
#             payload = hit.payload or {}
#             loc = payload.get('location', {})
#
#             # 内容展示逻辑
#             content_preview = payload.get('content', '')[
#                               :200] + "..." if result_type == "document" else f"{default_content} ({payload.get('year', 'Unknown')})"
#
#             item = SearchResultItem(
#                 id=str(hit.id),
#                 score=hit.score,
#                 year=payload.get('year', 0),
#                 lat=loc.get('lat', 0.0),
#                 lng=loc.get('lon', 0.0),
#                 source_dataset=payload.get('source_dataset') or payload.get('source_image') or 'Unknown',
#                 content=content_preview,
#                 fullData=payload,
#                 type=result_type,
#                 pixel_coords=payload.get('pixel_coords'),
#                 image_source=payload.get('source_image'),
#                 geo_polygon=payload.get('geo_detail')
#             )
#             results.append(item)
#         return results
#
#     # ==========================================================================
#     #  功能 1: 文本混合搜索 (Text -> Text & Image)
#     # ==========================================================================
#
# def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> \
#         List[SearchResultItem]:
#     """
#     实现逻辑：
#     1. 分别获取 Document (文搜文) 和 Map (文搜图) 结果。
#     2. 使用各自的“绝对阈值”过滤掉无关结果。
#     3. 对两组结果分别进行 Z-Score 归一化。
#     4. 合并结果。
#     5. 使用“相对阈值” (Z-Score > 0) 再次过滤，保留高于平均水平的结果。
#     6. 排序并返回。
#     """
#     client = GlobalState.get_db()
#     q_filter = self._build_qdrant_filters(filters)
#
#     # --- 配置参数 ---
#     DOC_MIN_SCORE = 0.35  # 文档绝对阈值 (MiniLM)
#     MAP_MIN_SCORE = 0.18  # 地图绝对阈值 (CLIP/PE)
#     Z_SCORE_THRESHOLD = -0.5  # 相对阈值 (标准差)，设为 0 表示只取平均分以上的，-0.5 表示稍宽容一点
#
#     doc_results = []
#     map_results = []
#
#     # 1. 搜文档 (MiniLM)
#     try:
#         text_model = GlobalState.get_text_model()
#         text_vec = text_model.encode(query).tolist()
#
#         hits_doc = client.query_points(
#             collection_name=self.DOC_COLLECTION,
#             query=text_vec,
#             using="text_vector",
#             query_filter=q_filter,
#             limit=limit * 2,  # 多取一倍用于后续筛选
#             with_payload=True
#         )
#         raw_docs = self._hits_to_results(hits_doc, "document")
#         # 🛡️ 绝对阈值过滤
#         doc_results = [r for r in raw_docs if r.score > DOC_MIN_SCORE]
#     except Exception as e:
#         print(f"⚠️ Doc search failed: {e}")
#
#     # 2. 搜地图 (PE/CLIP)
#     try:
#         pe_model = GlobalState.get_pe_model()
#         pe_vec = pe_model.extract_text_features(query)
#         if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
#         if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]
#
#         hits_map = client.query_points(
#             collection_name=self.MAP_COLLECTION,
#             query=pe_vec,
#             # maps 集合默认向量就是视觉向量
#             query_filter=q_filter,
#             limit=limit * 2,
#             with_payload=True
#         )
#         raw_maps = self._hits_to_results(hits_map, "map_tile", "Map Fragment")
#         # 🛡️ 绝对阈值过滤
#         map_results = [r for r in raw_maps if r.score > MAP_MIN_SCORE]
#     except Exception as e:
#         print(f"⚠️ Map search failed: {e}")
#
#     # --- 3. 独立归一化 (关键步骤) ---
#     # 必须分开归一化，因为两个模型的原始分数分布完全不同
#     if doc_results:
#         doc_results = self._normalize_scores(doc_results)
#
#     if map_results:
#         map_results = self._normalize_scores(map_results)
#
#     # --- 4. 合并与最终排序 ---
#     all_results = doc_results + map_results
#
#     # 🛡️ 相对阈值过滤 (Z-Score 过滤)
#     # 这一步是为了剔除在各自模型中表现都很差的“长尾”结果
#     final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]
#
#     # 排序
#     final_results.sort(key=lambda x: x.score, reverse=True)
#
#     return final_results[:limit]
#
#     # ==========================================================================
#     #  功能 2: 图片混合搜索 (Image -> Image & Text)
#     # ==========================================================================
#
# def search_image(self, image_data: bytes, limit: int, threshold: float) -> List[SearchResultItem]:
#     """
#     图片搜索同样应用 Z-Score 逻辑
#     """
#     client = GlobalState.get_db()
#     pe_model = GlobalState.get_pe_model()
#
#     try:
#         image = Image.open(io.BytesIO(image_data))
#         vector_list = pe_model.extract_image_features([image])[0].tolist()
#     except Exception as e:
#         raise ValueError(f"Invalid image: {e}")
#
#     # 图片搜索通常置信度较高，阈值可以高一点
#     MAP_IMG_MIN_SCORE = 0.45
#     DOC_IMG_MIN_SCORE = 0.40
#     Z_SCORE_THRESHOLD = -1.0  # 图片搜索结果较少，稍微宽容一点
#
#     doc_results = []
#     map_results = []
#
#     # 1. 搜地图 (图搜图)
#     try:
#         hits_map = client.query_points(
#             collection_name=self.MAP_COLLECTION,
#             query=vector_list,
#             limit=limit * 2,
#             with_payload=True
#         )
#         raw_maps = self._hits_to_results(hits_map, "map_tile", "Visual Match")
#         map_results = [r for r in raw_maps if r.score > MAP_IMG_MIN_SCORE]
#     except Exception as e:
#         print(f"⚠️ Image->Map search failed: {e}")
#
#     # 2. 搜文档 (图搜文 - 需文档库有 pe_vector)
#     try:
#         hits_doc = client.query_points(
#             collection_name=self.DOC_COLLECTION,
#             query=vector_list,
#             using="pe_vector",
#             limit=limit * 2,
#             with_payload=True
#         )
#         raw_docs = self._hits_to_results(hits_doc, "document")
#         doc_results = [r for r in raw_docs if r.score > DOC_IMG_MIN_SCORE]
#     except Exception as e:
#         print(f"⚠️ Image->Doc search failed: {e}")
#
#     # 3. 归一化与合并
#     if map_results: map_results = self._normalize_scores(map_results)
#     if doc_results: doc_results = self._normalize_scores(doc_results)
#
#     all_results = map_results + doc_results
#     final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]
#
#     final_results.sort(key=lambda x: x.score, reverse=True)
#
#     return final_results[:limit]
#
#     # ==========================================================================
#     #  功能 3: 3D 热力图数据
#     # ==========================================================================
#
# def get_heatmap_data(self, query: str, limit: int = 2000) -> List[HeatmapPoint]:
#     """
#     为了性能，热力图数据不进行复杂的归一化，直接返回原始分数即可，
#     或者只进行简单的 Min-Max 缩放。这里保持原始分数。
#     """
#     client = GlobalState.get_db()
#     payload_selector = models.PayloadSelectorInclude(include=["location"])
#     points = []
#
#     # 搜文档
#     try:
#         text_model = GlobalState.get_text_model()
#         vec = text_model.encode(query).tolist()
#         hits = client.query_points(
#             self.DOC_COLLECTION, query=vec, using="text_vector",
#             limit=limit // 2, with_payload=payload_selector, score_threshold=0.35
#         )
#         if hasattr(hits, 'points'): hits = hits.points
#         for h in hits:
#             loc = h.payload.get('location')
#             if loc: points.append(HeatmapPoint(lat=loc['lat'], lng=loc['lon'], score=h.score))
#     except:
#         pass
#
#     # 搜地图
#     try:
#         pe_model = GlobalState.get_pe_model()
#         vec = pe_model.extract_text_features(query)[0].tolist()
#         hits = client.query_points(
#             self.MAP_COLLECTION, query=vec,
#             limit=limit // 2, with_payload=payload_selector, score_threshold=0.20
#         )
#         if hasattr(hits, 'points'): hits = hits.points
#         for h in hits:
#             loc = h.payload.get('location')
#             if not loc:
#                 # 如果没有直接的 loc，尝试从 geo_detail 获取中心点 (如果有)
#                 pass
#             if loc: points.append(HeatmapPoint(lat=loc['lat'], lng=loc['lon'], score=h.score * 1.1))
#     except:
#         pass
#
#     return points
#
# # ==========================================================================
# #  功能 3: 热力图 (Heatmap)
# # ==========================================================================
#
# def get_heatmap_data(self, query: str, limit: int = 2000) -> List[HeatmapPoint]:
#     """
#     获取轻量级热力图数据
#     """
#     client = GlobalState.get_db()
#     points = []
#
#     # 仅包含位置信息的 Payload 筛选器 (高性能)
#     payload_selector = models.PayloadSelectorInclude(include=["location", "geo_detail"])
#
#     # 1. 搜文档 (MiniLM)
#     try:
#         text_model = GlobalState.get_text_model()
#         text_vec = text_model.encode(query).tolist()
#
#         hits = client.query_points(
#             collection_name=self.DOC_COLLECTION,
#             query=text_vec,
#             using="text_vector",
#             limit=limit // 2,
#             with_payload=payload_selector,
#             score_threshold=0.35
#         )
#         if hasattr(hits, 'points'): hits = hits.points
#
#         for hit in hits:
#             loc = hit.payload.get('location')
#             if loc and 'lat' in loc:
#                 points.append(HeatmapPoint(lat=loc['lat'], lng=loc['lon'], score=hit.score))
#     except Exception as e:
#         print(f"Heatmap doc error: {e}")
#
#     # 2. 搜地图 (PE)
#     try:
#         pe_model = GlobalState.get_pe_model()
#         pe_vec = pe_model.extract_text_features(query)
#         if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
#         if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]
#
#         hits = client.query_points(
#             collection_name=self.MAP_COLLECTION,
#             query=pe_vec,
#             limit=limit // 2,
#             with_payload=payload_selector,
#             score_threshold=0.15
#         )
#         if hasattr(hits, 'points'): hits = hits.points
#
#         for hit in hits:
#             loc = hit.payload.get('location')
#             # Fallback: 如果 location 空，尝试从 geo_detail 算
#             if not loc:
#                 geo = hit.payload.get('geo_detail', {}).get('wgs84', {})
#                 if 'center' in geo:
#                     loc = {'lat': geo['center'][0], 'lon': geo['center'][1]}
#
#             if loc and 'lat' in loc:
#                 # 地图结果加权 1.2
#                 points.append(HeatmapPoint(lat=loc['lat'], lng=loc['lon'], score=hit.score * 1.2))
#
#     except Exception as e:
#         print(f"Heatmap map error: {e}")
#
#     return points


import io
from typing import Optional, List

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.schema.search import SearchResultItem, SearchFilters, HeatmapPoint
from backend.app.utils.global_state import GlobalState

from qdrant_client import models


class SearchService:
    def __init__(self):
        self.MAP_COLLECTION = settings.MAP_COLLECTION
        self.DOC_COLLECTION = settings.DOC_COLLECTION

    # ==========================================================================
    #  核心算法: 归一化与辅助函数
    # ==========================================================================

    def _normalize_scores(self, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """
        Z-Score 归一化 (Standardization)
        公式: z = (x - μ) / σ
        作用: 将不同模型的分数映射到同一个标准正态分布上，使它们可以相互比较。
        """
        if not results or len(results) < 2:
            return results

        # 1. 提取分数
        scores = [r.score for r in results]
        mean = np.mean(scores)
        std = np.std(scores)

        # 2. 防御性处理：如果标准差为0 (所有分数都一样)，无法归一化
        if std == 0:
            return results

        # 3. 执行归一化
        for r in results:
            r.score = (r.score - mean) / std

        return results

    def _build_qdrant_filters(self, filters: SearchFilters) -> Optional[models.Filter]:
        """构建 Qdrant 过滤器"""
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

    def _hits_to_results(self, hits, result_type: str, default_content: str = "") -> List[SearchResultItem]:
        """将 Qdrant 返回的原始 hits 转换为统一的数据结构"""
        results = []
        if isinstance(hits, tuple): hits = hits[0]
        if hasattr(hits, 'points'): hits = hits.points
        if not hits: return results

        for hit in hits:
            if isinstance(hit, tuple) or not hasattr(hit, 'score'): continue

            payload = hit.payload or {}
            loc = payload.get('location', {})

            # 内容展示逻辑
            content_preview = payload.get('content', '')[
                              :200] + "..." if result_type == "document" else f"{default_content} ({payload.get('year', 'Unknown')})"

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
                image_source=payload.get('source_image'),
                geo_polygon=payload.get('geo_detail')
            )
            print(item)
            results.append(item)
        return results

    # ==========================================================================
    #  功能 1: 文本混合搜索 (Text -> Text & Image)
    # ==========================================================================

    def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
        SearchResultItem]:
        """
        实现逻辑：
        1. 分别获取 Document (文搜文) 和 Map (文搜图) 结果。
        2. 使用各自的“绝对阈值”过滤掉无关结果。
        3. 对两组结果分别进行 Z-Score 归一化。
        4. 合并结果。
        5. 使用“相对阈值” (Z-Score > 0) 再次过滤，保留高于平均水平的结果。
        6. 排序并返回。
        """
        client = GlobalState.get_db()
        q_filter = self._build_qdrant_filters(filters)

        # --- 配置参数 ---
        DOC_MIN_SCORE = 0.45  # 文档绝对阈值 (MiniLM)
        MAP_MIN_SCORE = 0.18  # 地图绝对阈值 (CLIP/PE)
        Z_SCORE_THRESHOLD = 0  # 相对阈值 (标准差)，设为 0 表示只取平均分以上的，-0.5 表示稍宽容一点

        doc_results = []
        map_results = []

        # 1. 搜文档 (MiniLM)
        try:
            text_model = GlobalState.get_text_model()
            text_vec = text_model.encode(query).tolist()

            hits_doc = client.query_points(
                collection_name=self.DOC_COLLECTION,
                query=text_vec,
                using="text_vector",
                query_filter=q_filter,
                limit=limit * 2,  # 多取一倍用于后续筛选
                with_payload=True
            )
            raw_docs = self._hits_to_results(hits_doc, "document")
            # 🛡️ 绝对阈值过滤
            doc_results = [r for r in raw_docs if r.score > DOC_MIN_SCORE]
        except Exception as e:
            print(f"⚠️ Doc search failed: {e}")

        # 2. 搜地图 (PE/CLIP)
        try:
            pe_model = GlobalState.get_pe_model()
            pe_vec = pe_model.extract_text_features(query)
            if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
            if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]

            hits_map = client.query_points(
                collection_name=self.MAP_COLLECTION,
                query=pe_vec,
                # maps 集合默认向量就是视觉向量
                query_filter=q_filter,
                limit=limit * 2,
                with_payload=True
            )
            raw_maps = self._hits_to_results(hits_map, "map_tile", "Map Fragment")
            # 🛡️ 绝对阈值过滤
            map_results = [r for r in raw_maps if r.score > MAP_MIN_SCORE]
        except Exception as e:
            print(f"⚠️ Map search failed: {e}")

        # --- 3. 独立归一化 (关键步骤) ---
        # 必须分开归一化，因为两个模型的原始分数分布完全不同
        if doc_results:
            doc_results = self._normalize_scores(doc_results)

        if map_results:
            map_results = self._normalize_scores(map_results)

        # --- 4. 合并与最终排序 ---
        all_results = doc_results + map_results

        # 🛡️ 相对阈值过滤 (Z-Score 过滤)
        # 这一步是为了剔除在各自模型中表现都很差的“长尾”结果
        final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]

        # 排序
        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:limit]

    # ==========================================================================
    #  功能 2: 图片混合搜索 (Image -> Image & Text)
    # ==========================================================================

    def search_image(self, image_data: bytes, limit: int, threshold: float) -> List[SearchResultItem]:
        """
        图片搜索同样应用 Z-Score 逻辑
        """
        client = GlobalState.get_db()
        pe_model = GlobalState.get_pe_model()

        try:
            image = Image.open(io.BytesIO(image_data))
            vector_list = pe_model.extract_image_features([image])[0].tolist()
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")

        # 图片搜索通常置信度较高，阈值可以高一点
        MAP_IMG_MIN_SCORE = 0.40
        DOC_IMG_MIN_SCORE = 0.22
        Z_SCORE_THRESHOLD = 0  # 图片搜索结果较少，稍微宽容一点

        doc_results = []
        map_results = []

        # 1. 搜地图 (图搜图)
        try:
            hits_map = client.query_points(
                collection_name=self.MAP_COLLECTION,
                query=vector_list,
                limit=limit * 2,
                with_payload=True
            )
            raw_maps = self._hits_to_results(hits_map, "map_tile", "Visual Match")
            map_results = [r for r in raw_maps if r.score > MAP_IMG_MIN_SCORE]
        except Exception as e:
            print(f"⚠️ Image->Map search failed: {e}")

        # 2. 搜文档 (图搜文 - 需文档库有 pe_vector)
        try:
            hits_doc = client.query_points(
                collection_name=self.DOC_COLLECTION,
                query=vector_list,
                using="pe_vector",
                limit=limit * 2,
                with_payload=True
            )

            raw_docs = self._hits_to_results(hits_doc, "document")
            doc_results = [r for r in raw_docs if r.score > DOC_IMG_MIN_SCORE]
        except Exception as e:
            print(f"⚠️ Image->Doc search failed: {e}")

        # 3. 归一化与合并
        if map_results: map_results = self._normalize_scores(map_results)
        if doc_results: doc_results = self._normalize_scores(doc_results)

        all_results = map_results + doc_results
        final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]

        final_results.sort(key=lambda x: x.score, reverse=True)

        return final_results[:limit]

    # ==========================================================================
    #  功能 3: 3D 热力图数据
    # ==========================================================================

    def get_heatmap_data(self, query: str, limit: int = 2000) -> List[HeatmapPoint]:
        """
        为了性能，热力图数据不进行复杂的归一化，直接返回原始分数即可，
        或者只进行简单的 Min-Max 缩放。这里保持原始分数。
        """
        client = GlobalState.get_db()
        payload_selector = models.PayloadSelectorInclude(include=["location"])
        points = []

        # 搜文档
        try:
            text_model = GlobalState.get_text_model()
            vec = text_model.encode(query).tolist()
            hits = client.query_points(
                self.DOC_COLLECTION, query=vec, using="text_vector",
                limit=limit // 2, with_payload=payload_selector, score_threshold=0.35
            )
            if hasattr(hits, 'points'): hits = hits.points
            for h in hits:
                loc = h.payload.get('location')
                if loc: points.append(HeatmapPoint(lat=loc['lat'], lng=loc['lon'], score=h.score))
        except:
            pass

        # 搜地图
        try:
            pe_model = GlobalState.get_pe_model()
            vec = pe_model.extract_text_features(query)[0].tolist()
            hits = client.query_points(
                self.MAP_COLLECTION, query=vec,
                limit=limit // 2, with_payload=payload_selector, score_threshold=0.20
            )
            if hasattr(hits, 'points'): hits = hits.points
            for h in hits:
                loc = h.payload.get('location')
                if not loc:
                    # 如果没有直接的 loc，尝试从 geo_detail 获取中心点 (如果有)
                    pass
                if loc: points.append(HeatmapPoint(lat=loc['lat'], lng=loc['lon'], score=h.score * 1.1))
        except:
            pass

        return points


# 导出单例
search_service = SearchService()
