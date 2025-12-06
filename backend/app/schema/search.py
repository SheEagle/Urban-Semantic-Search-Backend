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
    pixel_coords: List[int]
    geo_polygon: Optional[Dict[str, Any]] = None
    image_source: Optional[str] = None


class SearchResponse(BaseModel):
    status: str
    count: int
    data: List[SearchResultItem]
