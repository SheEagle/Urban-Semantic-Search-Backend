from pydantic import BaseModel
from typing import List, Optional, Any, Dict


class TextSearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.2


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
