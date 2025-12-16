import struct
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import Response

from backend.app.schema.search import TextSearchRequest, SearchResponse, HeatmapResponse
from backend.app.service.search_service import search_service

# Configure Logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    Hybrid Text Search: Searches both Documents (Semantic) and Map Tiles (Text-Image matching).
    """
    try:
        results = search_service.search_text(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold,
            filters=request.filters
        )

        return SearchResponse(
            status="success",
            count=len(results),
            data=results
        )
    except Exception as e:
        logger.error(f"Text Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
        file: UploadFile = File(...),
        limit: int = Form(20),  # Received as Form Data
        threshold: float = Form(0.2)  # Received as Form Data
):
    """
    Hybrid Image Search: Upload image -> Find visually similar Map Tiles + Contextually relevant Documents.
    """
    try:
        image_bytes = await file.read()

        results = search_service.search_image(
            image_data=image_bytes,
            limit=limit,
            threshold=threshold
        )

        return SearchResponse(
            status="success",
            count=len(results),
            data=results
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Image Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heatmap/binary")
async def get_heatmap_binary(limit: int = 10000):
    """
    Returns heatmap data in binary format for extreme performance (optional usage).
    Format: Each point consists of 3 float32s (lat, lng, score) -> 12 bytes per point.
    10,000 points take only ~120KB, which parses significantly faster than JSON.
    """
    try:
        points = search_service.get_heatmap_points(limit=limit)  # Returns list of dicts

        byte_array = bytearray()
        for p in points:
            # 'fff' stands for 3 floats
            byte_array.extend(struct.pack('fff', p['lat'], p['lng'], p['score']))

        return Response(content=bytes(byte_array), media_type="application/octet-stream")
    except Exception as e:
        logger.error(f"Binary Heatmap Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate binary data")


@router.get("/heatmap-data", response_model=HeatmapResponse)
async def get_heatmap_data(
        query: str = Query(None,
                           description="Optional search query to generate heatmap relevance. If empty, returns general density."),
        limit: int = Query(2000, description="Max points to return")
):
    """
    High-performance endpoint designed specifically for the 3D DeckGL view.
    Returns only coordinates and relevance scores, excluding heavy metadata.
    """
    # Cap the limit to prevent frontend performance issues
    safe_limit = min(limit, 5000)

    try:
        points = search_service.get_heatmap_data(query, safe_limit)

        return HeatmapResponse(
            status="success",
            count=len(points),
            data=points
        )
    except Exception as e:
        logger.error(f"Heatmap Data Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
