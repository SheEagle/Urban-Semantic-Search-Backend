from typing import List, Optional, Any, Dict
from pydantic import BaseModel


# --- 1. Filter Models ---

class SearchFilters(BaseModel):
    """
    Filters applied to search queries.
    """
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    map_source: Optional[str] = None
    # Geographic bounding box format: [min_lon, min_lat, max_lon, max_lat]
    geo_bbox: Optional[List[float]] = None


class TextSearchRequest(BaseModel):
    """
    Payload for text-based search requests.
    """
    query: str
    limit: int = 20
    threshold: float = 0.2
    filters: Optional[SearchFilters] = None


# --- 2. Search Result Models ---

class SearchResultItem(BaseModel):
    """
    Represents a single search result item (either a Map Tile or a Document).
    """
    id: str
    score: float
    lat: float
    lng: float
    year: int

    # Map-specific fields
    pixel_coords: Optional[List[int]] = None
    image_source: Optional[str] = None

    # Metadata fields
    source_dataset: Optional[str] = None
    content: Optional[str] = None  # Content summary for documents or title for maps
    type: str  # Type discriminator: 'map_tile' or 'document'
    fullData: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """
    Standard wrapper for search responses.
    """
    status: str
    count: int
    data: List[SearchResultItem]


# --- 3. Heatmap Models ---

class HeatmapPoint(BaseModel):
    """
    Lightweight data point for 3D visualizations.
    """
    lat: float
    lng: float
    # Default is 1.0 for density mode; represents similarity score if a search query is present
    score: float = 1.0


class HeatmapResponse(BaseModel):
    """
    Wrapper for heatmap data responses.
    """
    status: str
    count: int
    data: List[HeatmapPoint]
