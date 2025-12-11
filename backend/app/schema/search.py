from pydantic import BaseModel
from typing import List, Optional, Any, Dict


# 1. 定义过滤器模型
class SearchFilters(BaseModel):
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    map_source: Optional[str] = None
    # 坐标范围通常是一个矩形框 (min_lon, min_lat, max_lon, max_lat)
    geo_bbox: Optional[List[float]] = None


class TextSearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.2
    filters: Optional[SearchFilters] = None  # 新增字段


class SearchResultItem(BaseModel):
    id: str
    score: float
    lat: float
    lng: float
    year: int
    pixel_coords: Optional[List[int]] = None  # 地图特有
    image_source: Optional[str] = None  # 来源名称
    source_dataset: Optional[str] = None
    content: Optional[str] = None  # 🔥 新增: 文档内容摘要 / 地图标题
    type: str  # 🔥 新增: 'map_tile' 或 'document'
    fullData: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    status: str
    count: int
    data: List[SearchResultItem]


from pydantic import BaseModel
from typing import List, Optional


class HeatmapPoint(BaseModel):
    lat: float
    lng: float
    score: float = 1.0  # 默认为1，如果有搜索词则为相似度


class HeatmapResponse(BaseModel):
    status: str
    count: int
    data: List[HeatmapPoint]
