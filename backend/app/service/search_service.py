import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.repository.qdrant_repo import QdrantRepository
from backend.app.schema.search import SearchResultItem, SearchFilters, HeatmapPoint
from backend.app.utils.global_state import GlobalState

from qdrant_client import models


class SearchService:
    def __init__(self):
        self.MAP_COLLECTION = settings.MAP_COLLECTION
        self.DOC_COLLECTION = settings.DOC_COLLECTION
        self.repo = QdrantRepository()

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
            # print(item)
            results.append(item)
        return results

    # ==========================================================================
    #  功能 1: 文本混合搜索 (Text -> Text & Image)
    # ==========================================================================

    # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
    #     SearchResultItem]:
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
    #     DOC_MIN_SCORE = 0.45  # 文档绝对阈值 (MiniLM)
    #     MAP_MIN_SCORE = 0.18  # 地图绝对阈值 (CLIP/PE)
    #     Z_SCORE_THRESHOLD = 0  # 相对阈值 (标准差)，设为 0 表示只取平均分以上的，-0.5 表示稍宽容一点
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
    # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
    #     SearchResultItem]:
    #     t0 = time.time()
    #     timings = {}
    #
    #     def log_time(step_name, start_time):
    #         elapsed = time.time() - start_time
    #         timings[step_name] = elapsed
    #         print(f"⏱️ [Step: {step_name}] 耗时: {elapsed:.4f}s")
    #
    #     # --- 1. 准备工作 ---
    #     client = GlobalState.get_db()
    #     q_filter = self._build_qdrant_filters(filters)
    #
    #     DOC_MIN_SCORE = 0.45
    #     MAP_MIN_SCORE = 0.18
    #     Z_SCORE_THRESHOLD = 0
    #
    #     doc_results = []
    #     map_results = []
    #
    #     # --- 2. 搜文档流程 ---
    #     t_doc_start = time.time()
    #
    #     # 2.1 文本向量化 (模型推理)
    #     t_model_doc = time.time()
    #     try:
    #         text_model = GlobalState.get_text_model()
    #         text_vec = text_model.encode(query).tolist()
    #     except Exception as e:
    #         text_vec = []
    #         print(f"Doc Model Error: {e}")
    #     log_time("Doc Embedding (模型)", t_model_doc)
    #
    #     # 2.2 文档检索 (Qdrant 网络 I/O)
    #     t_search_doc = time.time()
    #     try:
    #         if text_vec:
    #             hits_doc = client.query_points(
    #                 collection_name=self.DOC_COLLECTION,
    #                 query=text_vec,
    #                 using="text_vector",
    #                 query_filter=q_filter,
    #                 limit=limit * 2,
    #                 with_payload=True
    #             )
    #             raw_docs = self._hits_to_results(hits_doc, "document")
    #             doc_results = [r for r in raw_docs if r.score > DOC_MIN_SCORE]
    #     except Exception as e:
    #         print(f"Doc Search Error: {e}")
    #     log_time("Doc Qdrant Search (IO)", t_search_doc)
    #
    #     log_time("--> 文档搜索总耗时", t_doc_start)
    #     print("-" * 30)
    #
    #     # --- 3. 搜地图流程 ---
    #     t_map_start = time.time()
    #
    #     # 3.1 地图向量化 (模型推理)
    #     t_model_map = time.time()
    #     try:
    #         pe_model = GlobalState.get_pe_model()
    #         pe_vec = pe_model.extract_text_features(query)
    #         if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
    #         if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]
    #     except Exception as e:
    #         pe_vec = []
    #         print(f"Map Model Error: {e}")
    #     log_time("Map Embedding (模型)", t_model_map)
    #
    #     # 3.2 地图检索 (Qdrant 网络 I/O)
    #     t_search_map = time.time()
    #     try:
    #         if pe_vec:
    #             hits_map = client.query_points(
    #                 collection_name=self.MAP_COLLECTION,
    #                 query=pe_vec,
    #                 query_filter=q_filter,
    #                 limit=limit * 2,
    #                 with_payload=True
    #             )
    #             raw_maps = self._hits_to_results(hits_map, "map_tile", "Map Fragment")
    #             map_results = [r for r in raw_maps if r.score > MAP_MIN_SCORE]
    #     except Exception as e:
    #         print(f"Map Search Error: {e}")
    #     log_time("Map Qdrant Search (IO)", t_search_map)
    #
    #     log_time("--> 地图搜索总耗时", t_map_start)
    #     print("-" * 30)
    #
    #     # --- 4. 归一化与后处理 ---
    #     t_process = time.time()
    #     if doc_results: doc_results = self._normalize_scores(doc_results)
    #     if map_results: map_results = self._normalize_scores(map_results)
    #     all_results = doc_results + map_results
    #     final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]
    #     final_results.sort(key=lambda x: x.score, reverse=True)
    #     log_time("Normalization & Sort", t_process)
    #
    #     # --- 总结报告 ---
    #     total_time = time.time() - t0
    #     print("\n" + "=" * 40)
    #     print(f"📊 性能诊断报告 (总耗时: {total_time:.4f}s)")
    #     print(f"1. 文本模型计算: {timings.get('Doc Embedding (模型)', 0):.4f}s")
    #     print(f"2. 文档 IO 耗时: {timings.get('Doc Qdrant Search (IO)', 0):.4f}s")
    #     print(f"3. 地图模型计算: {timings.get('Map Embedding (模型)', 0):.4f}s")
    #     print(f"4. 地图 IO 耗时: {timings.get('Map Qdrant Search (IO)', 0):.4f}s")
    #     print("=" * 40 + "\n")
    #
    #     return final_results[:limit]

    # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
    #     SearchResultItem]:
    #     """
    #     极速优化版：并行执行 + Payload 瘦身 + 服务端过滤
    #     """
    #     client = GlobalState.get_db()
    #     q_filter = self._build_qdrant_filters(filters)
    #
    #     # --- 配置参数 ---
    #     DOC_MIN_SCORE = 0.45
    #     MAP_MIN_SCORE = 0.18
    #     Z_SCORE_THRESHOLD = 0
    #
    #     # 关键优化 1: 定义轻量级 Payload 过滤器
    #     # 排除掉那些巨大的字段，只取列表展示需要的核心字段
    #     # 如果前端点击详情需要完整数据，建议用 ID 再去调一次 retrieve 接口，而不是在搜索列表里全拉回来
    #     # payload_selector = models.PayloadSelectorInclude(
    #     #     include=["year", "location", "source_dataset", "source_image", "content"]  # 注意：根据情况 content 也可以截断或排除
    #     # )
    #     # 或者使用 Exclude 模式 (推荐，更安全):
    #     payload_selector = models.PayloadSelectorExclude(
    #         exclude=["geo_detail", "full_metadata", "pixel_coords"]
    #     )
    #
    #     def fetch_docs():
    #         try:
    #             # [优化建议] 如果可能，请将 GlobalState 中的模型移至 GPU: model.to('cuda')
    #             text_model = GlobalState.get_text_model()
    #             text_vec = text_model.encode(query).tolist()
    #
    #             hits = client.query_points(
    #                 collection_name=self.DOC_COLLECTION,
    #                 query=text_vec,
    #                 using="text_vector",
    #                 query_filter=q_filter,
    #                 limit=limit * 2,
    #                 with_payload=True,  # 🚀 瘦身：只拉取必要字段
    #                 score_threshold=DOC_MIN_SCORE,  # 🚀 过滤：DB端直接过滤低分
    #                 search_params=models.SearchParams(
    #                     hnsw_ef=32,  # 默认可能是 null (自动) 或较高。
    #                     # 调低这个值 (比如 64 或 32) 会显著提速，但会略微降低长尾结果的召回率。
    #                     exact=False  # 确保关闭精确搜索
    #                 )
    #             )
    #             return self._hits_to_results(hits, "document")
    #         except Exception as e:
    #             print(f"⚠️ Doc Error: {e}")
    #             return []
    #
    #     def fetch_maps():
    #         try:
    #             pe_model = GlobalState.get_pe_model()
    #             pe_vec = pe_model.extract_text_features(query)
    #             if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
    #             if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]
    #
    #             hits = client.query_points(
    #                 collection_name=self.MAP_COLLECTION,
    #                 query=pe_vec,
    #                 query_filter=q_filter,
    #                 limit=limit * 2,
    #                 with_payload=True,  # 🚀 瘦身
    #                 score_threshold=MAP_MIN_SCORE,  # 🚀 过滤
    #                 search_params=models.SearchParams(
    #                     hnsw_ef=128,  # 默认可能是 null (自动) 或较高。
    #                     # 调低这个值 (比如 64 或 32) 会显著提速，但会略微降低长尾结果的召回率。
    #                     exact=False  # 确保关闭精确搜索
    #                 )
    #             )
    #             return self._hits_to_results(hits, "map_tile", "Map Fragment")
    #         except Exception as e:
    #             print(f"⚠️ Map Error: {e}")
    #             return []
    #
    #     # 关键优化 2: 并行执行 (ThreadPool)
    #     # 之前是串行：0.9 + 2.3 + 0.9 + 2.1 = 6.2s
    #     # 现在是并行：Max(Doc流程, Map流程)
    #     doc_results, map_results = [], []
    #
    #     t_start = time.time()
    #     with ThreadPoolExecutor(max_workers=2) as executor:
    #         future_doc = executor.submit(fetch_docs)
    #         future_map = executor.submit(fetch_maps)
    #
    #         # 等待结果
    #         doc_results = future_doc.result()
    #         map_results = future_map.result()
    #
    #     print(f"⚡ Search completed in {time.time() - t_start:.4f}s")
    #
    #     # --- 归一化与合并 (逻辑不变) ---
    #     if doc_results: self._normalize_scores(doc_results)
    #     if map_results: self._normalize_scores(map_results)
    #
    #     all_results = doc_results + map_results
    #     final_results = [r for r in all_results if r.score > Z_SCORE_THRESHOLD]
    #     final_results.sort(key=lambda x: x.score, reverse=True)
    #
    #     return final_results[:limit]

    import time
    from concurrent.futures import ThreadPoolExecutor
    from qdrant_client import models

    # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
    #     SearchResultItem]:
    #     """
    #     极速优化 V2：修复 Payload Bug + 推理分离 + 参数调优
    #     """
    #     t_start = time.time()
    #     client = GlobalState.get_db()  # 确保这里返回的是同一个单例，不要每次重建连接
    #     q_filter = self._build_qdrant_filters(filters)
    #
    #     # --- 1. 配置参数调优 ---
    #     DOC_MIN_SCORE = 0.45
    #     MAP_MIN_SCORE = 0.18
    #     # 调低 hnsw_ef (搜索时的探索广度)。对于 Top-K 检索，16-32 通常足够，速度极快。
    #     SEARCH_PARAMS = models.SearchParams(hnsw_ef=32, exact=False)
    #
    #     # --- 2. 真正生效的 Payload 瘦身 ---
    #     # 定义好要包含或排除的字段
    #     payload_selector = models.PayloadSelectorExclude(
    #         exclude=["geo_detail", "full_metadata", "pixel_coords", "embedding"]  # 确保排除 embedding 本身，如果存储了的话
    #     )
    #
    #     # --- 3. 模型推理 (CPU/GPU 密集型) ---
    #     # 建议串行执行或移出此函数。如果模型很大，放在线程池里容易导致资源争抢或显存溢出。
    #     # 打印时间以定位瓶颈
    #     t_encode_start = time.time()
    #
    #     try:
    #         text_model = GlobalState.get_text_model()
    #         text_vec = text_model.encode(query).tolist()
    #     except Exception as e:
    #         print(f"⚠️ Text Model Error: {e}")
    #         return []
    #
    #     try:
    #         pe_model = GlobalState.get_pe_model()
    #         pe_vec = pe_model.extract_text_features(query)
    #         if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
    #         if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]
    #     except Exception as e:
    #         print(f"⚠️ PE Model Error: {e}")
    #         pe_vec = None
    #
    #     print(f"⏱️ Encoding time: {time.time() - t_encode_start:.4f}s")  # 👈 观察这里是否占据了 2.5s 中的大部分
    #
    #     # --- 4. 并行查询 (I/O 密集型) ---
    #     # 现在线程池里只做纯粹的网络 I/O，效率最高
    #
    #     def search_docs_io():
    #         if not text_vec: return []
    #         return client.query_points(
    #             collection_name=self.DOC_COLLECTION,
    #             query=text_vec,
    #             using="text_vector",
    #             query_filter=q_filter,
    #             limit=limit * 2,  # 稍微多取一点用于重排序
    #             with_payload=True,  # ✅ 关键修正：传入 selector
    #             score_threshold=DOC_MIN_SCORE,
    #             search_params=SEARCH_PARAMS
    #         )
    #
    #     def search_maps_io():
    #         if pe_vec is None: return []
    #         return client.query_points(
    #             collection_name=self.MAP_COLLECTION,
    #             query=pe_vec,
    #             query_filter=q_filter,
    #             limit=limit * 2,
    #             with_payload=True,  # ✅ 关键修正：传入 selector
    #             score_threshold=MAP_MIN_SCORE,
    #             search_params=SEARCH_PARAMS
    #         )
    #
    #     doc_hits, map_hits = [], []
    #
    #     # 因为只是发网络请求，开销极小，线程池才真正发挥作用
    #     t_search_start = time.time()
    #     with ThreadPoolExecutor(max_workers=2) as executor:
    #         future_doc = executor.submit(search_docs_io)
    #         future_map = executor.submit(search_maps_io)
    #
    #         try:
    #             doc_hits = future_doc.result()
    #         except Exception as e:
    #             print(f"⚠️ Doc Search Error: {e}")
    #
    #         try:
    #             map_hits = future_map.result()
    #         except Exception as e:
    #             print(f"⚠️ Map Search Error: {e}")
    #
    #     print(f"⏱️ Qdrant I/O time: {time.time() - t_search_start:.4f}s")
    #
    #     # --- 5. 结果处理 ---
    #     # (保持原有逻辑)
    #     doc_results = self._hits_to_results(doc_hits, "document")
    #     map_results = self._hits_to_results(map_hits, "map_tile", "Map Fragment")
    #
    #     if doc_results: self._normalize_scores(doc_results)
    #     if map_results: self._normalize_scores(map_results)
    #
    #     all_results = doc_results + map_results
    #     # Z_SCORE_THRESHOLD 建议设为全局变量
    #     final_results = [r for r in all_results if r.score > 0]
    #     final_results.sort(key=lambda x: x.score, reverse=True)
    #
    #     print(f"⚡ Total Search completed in {time.time() - t_start:.4f}s")
    #     return final_results[:limit]
    def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
        SearchResultItem]:
        """
        业务逻辑：文本混合搜索
        """
        t_start = time.time()

        # --- 1. 参数定义 ---
        DOC_MIN_SCORE = 0.45
        MAP_MIN_SCORE = 0.18

        # --- 2. 模型推理 (CPU/GPU 计算) ---
        t_encode = time.time()
        text_vec = []
        pe_vec = []

        try:
            # 假设这些 get_model 操作很快，或者你可以进一步封装 ModelService
            text_vec = GlobalState.get_text_model().encode(query).tolist()
        except Exception as e:
            print(f"Text Model Error: {e}")

        try:
            pe_raw = GlobalState.get_pe_model().extract_text_features(query)
            # 处理一下维度问题
            if hasattr(pe_raw, 'tolist'): pe_raw = pe_raw.tolist()
            if isinstance(pe_raw, list) and isinstance(pe_raw[0], list): pe_raw = pe_raw[0]
            pe_vec = pe_raw
        except Exception as e:
            print(f"PE Model Error: {e}")

        print(f"⏱️ Encoding: {time.time() - t_encode:.4f}s")

        # --- 3. 并行数据库查询 (IO 密集型) ---
        # 定义任务函数，直接调用 Repository
        def fetch_docs():
            if not text_vec: return []
            # 调用 Repository
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
            # 调用 Repository
            return self.repo.search(
                collection_name=self.MAP_COLLECTION,
                query_vector=pe_vec,
                filters=filters,
                limit=limit * 2,
                score_threshold=MAP_MIN_SCORE,
                hnsw_ef=32
            )

        # 执行并行
        t_search = time.time()
        doc_hits, map_hits = [], []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_doc = executor.submit(fetch_docs)
            future_map = executor.submit(fetch_maps)
            doc_hits = future_doc.result()
            map_hits = future_map.result()
        print(f"⏱️ IO Search: {time.time() - t_search:.4f}s")

        # --- 4. 结果转换与归一化 ---
        doc_results = self._hits_to_results(doc_hits, "document")
        map_results = self._hits_to_results(map_hits, "map_tile", "Map Fragment")

        if doc_results: self._normalize_scores(doc_results)
        if map_results: self._normalize_scores(map_results)

        # --- 5. 合并与排序 ---
        all_results = doc_results + map_results
        # 0 表示只取高于平均分的
        final_results = [r for r in all_results if r.score > 0]
        final_results.sort(key=lambda x: x.score, reverse=True)

        print(f"⚡ Total: {time.time() - t_start:.4f}s")
        return final_results[:limit]

    #
    # # ==========================================================================
    # #  功能 2: 图片混合搜索 (Image -> Image & Text)
    # # ==========================================================================
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
    #     MAP_IMG_MIN_SCORE = 0.40
    #     DOC_IMG_MIN_SCORE = 0.22
    #     Z_SCORE_THRESHOLD = 0  # 图片搜索结果较少，稍微宽容一点
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
    #
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
    def search_image(self, image_data: bytes, limit: int, threshold: float) -> List[SearchResultItem]:
        """
        图片混合搜索 (Image -> Image & Text)
        重构版：使用 Repository 层 + 并行查询
        """
        t_start = time.time()

        # --- 1. 参数定义 ---
        # 图片搜索通常置信度较高，阈值可以设高一点
        MAP_IMG_MIN_SCORE = 0.40
        DOC_IMG_MIN_SCORE = 0.22


        # --- 2. 图像特征提取 (CPU 密集) ---
        t_encode = time.time()
        try:
            image = Image.open(io.BytesIO(image_data))
            pe_model = GlobalState.get_pe_model()
            # 提取向量并转为 list
            vector_list = pe_model.extract_image_features([image])[0].tolist()
        except Exception as e:
            # 图片无效直接报错或返回空，视业务需求定
            print(f"⚠️ Image Encoding Error: {e}")
            raise ValueError(f"Invalid image processing: {e}")
        print(f"⏱️ Image Encoding: {time.time() - t_encode:.4f}s")

        # --- 3. 并行数据库查询 (IO 密集) ---
        def fetch_maps():
            # 图搜图 (Visual Match)
            return self.repo.search(
                collection_name=self.MAP_COLLECTION,
                query_vector=vector_list,
                limit=limit * 2,
                score_threshold=MAP_IMG_MIN_SCORE,
                hnsw_ef=32
            )

        def fetch_docs():
            # 图搜文 (Visual -> Text Description)
            # 注意：必须指定使用 "pe_vector" (视觉对齐向量)
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

        # 并行执行
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_map = executor.submit(fetch_maps)
            future_doc = executor.submit(fetch_docs)

            map_hits = future_map.result()
            doc_hits = future_doc.result()
        print(f"⏱️ IO Search: {time.time() - t_search:.4f}s")

        # --- 4. 结果转换 ---
        # 注意类型标识：map_tile vs document
        map_results = self._hits_to_results(map_hits, "map_tile", "Visual Match")
        doc_results = self._hits_to_results(doc_hits, "document")

        # --- 5. 归一化 (Normalization) ---
        if map_results: self._normalize_scores(map_results)
        if doc_results: self._normalize_scores(doc_results)

        # --- 6. 合并与排序 ---
        all_results = map_results + doc_results

        # Z_SCORE_THRESHOLD = 0 (取平均分以上)
        final_results = [r for r in all_results if r.score > 0]
        final_results.sort(key=lambda x: x.score, reverse=True)

        print(f"⚡ Image Search Total: {time.time() - t_start:.4f}s")
        return final_results[:limit]

    # def search_text(self, query: str, limit: int, threshold: float, filters: Optional[SearchFilters] = None) -> List[
    #     SearchResultItem]:
    #     print("\n🕵️‍♀️ [开始诊断] 3秒性能瓶颈分析...")
    #     start_time_all = time.time()
    #
    #     # 用于记录日志的列表，最后统一打印，避免print阻塞影响测速
    #     timeline = []
    #
    #     def record(thread_name, action, start_t):
    #         duration = time.time() - start_t
    #         timeline.append(f"[{time.time() - start_time_all:.3f}s] 🧵 {thread_name}: {action} 耗时 {duration:.3f}s")
    #         return duration
    #
    #     client = GlobalState.get_db()
    #     q_filter = self._build_qdrant_filters(filters)
    #
    #     # --- 任务 A: 搜文档 ---
    #     def task_docs():
    #         t_start = time.time()
    #
    #         # 1. 模型 Embedding
    #         t0 = time.time()
    #         try:
    #             text_model = GlobalState.get_text_model()
    #             text_vec = text_model.encode(query).tolist()
    #             record("Doc线程", "🧠模型计算", t0)
    #         except Exception:
    #             text_vec = []
    #
    #         # 2. Qdrant 搜索
    #         t1 = time.time()
    #         try:
    #             hits = client.query_points(
    #                 collection_name=self.DOC_COLLECTION,
    #                 query=text_vec,
    #                 using="text_vector",
    #                 query_filter=q_filter,
    #                 limit=limit * 2,
    #                 with_payload=True  # 👈 瓶颈嫌疑点：全量 Payload
    #             )
    #             res = self._hits_to_results(hits, "document")
    #             record("Doc线程", "☁️网络IO(Search)", t1)
    #             return res
    #         except Exception as e:
    #             print(f"Doc Error: {e}")
    #             return []
    #
    #     # --- 任务 B: 搜地图 ---
    #     def task_maps():
    #         t_start = time.time()
    #
    #         # 1. 模型 Embedding
    #         t0 = time.time()
    #         try:
    #             pe_model = GlobalState.get_pe_model()
    #             # 假设这里有一些处理逻辑
    #             pe_vec = pe_model.extract_text_features(query)
    #             if hasattr(pe_vec, 'tolist'): pe_vec = pe_vec.tolist()
    #             if isinstance(pe_vec, list) and isinstance(pe_vec[0], list): pe_vec = pe_vec[0]
    #             record("Map线程", "🧠模型计算", t0)
    #         except Exception:
    #             pe_vec = []
    #
    #         # 2. Qdrant 搜索
    #         t1 = time.time()
    #         try:
    #             hits = client.query_points(
    #                 collection_name=self.MAP_COLLECTION,
    #                 query=pe_vec,
    #                 query_filter=q_filter,
    #                 limit=limit * 2,
    #                 with_payload=True  # 👈 瓶颈嫌疑点：全量 Payload
    #             )
    #             res = self._hits_to_results(hits, "map_tile", "Map Fragment")
    #             record("Map线程", "☁️网络IO(Search)", t1)
    #             return res
    #         except Exception as e:
    #             print(f"Map Error: {e}")
    #             return []
    #
    #     # --- 并行执行 ---
    #     doc_results = []
    #     map_results = []
    #
    #     with ThreadPoolExecutor(max_workers=2) as executor:
    #         future_doc = executor.submit(task_docs)
    #         future_map = executor.submit(task_maps)
    #
    #         doc_results = future_doc.result()
    #         map_results = future_map.result()
    #
    #     total_time = time.time() - start_time_all
    #
    #     # --- 打印报告 ---
    #     print("\n" + "=" * 40)
    #     print("⏱️ 时间轴报告:")
    #     for log in sorted(timeline):  # 按时间排序
    #         print(log)
    #     print("-" * 40)
    #     print(f"📉 总耗时: {total_time:.3f}s")
    #     print("=" * 40 + "\n")
    #
    #     # (原本的后续处理逻辑，为了跑通暂且保留)
    #     all_results = doc_results + map_results
    #     return all_results[:limit]

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
