from fastapi import APIRouter, HTTPException, Depends, Query
from backend.app.schema.search import TextSearchRequest, SearchResponse, HeatmapResponse
from backend.app.service.search_service import search_service
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

router = APIRouter()


@router.post("/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    文本混合搜索：同时搜索 Document (语义) 和 Map Tile (图文匹配)
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
        print(f"❌ Text Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
        file: UploadFile = File(...),
        limit: int = Form(20),  # Form Data
        threshold: float = Form(0.2)  # Form Data
):
    """
    图片混合搜索：上传图片 -> 搜相似地图切片 + 搜相关文档
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
        print(f"❌ Image Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


import struct
from fastapi.responses import Response


@router.get("/heatmap/binary")
async def get_heatmap_binary(limit: int = 10000):
    points = search_service.get_heatmap_points(limit=limit)  # 获取 dict 列表

    # 格式：每个点由 3 个 float32 组成 (lat, lng, score) -> 12 bytes per point
    # 10000 点仅需 ~120KB，比 JSON 小得多且解析极快

    byte_array = bytearray()
    for p in points:
        # 'fff' 代表 3 个 float
        byte_array.extend(struct.pack('fff', p['lat'], p['lng'], p['score']))

    return Response(content=bytes(byte_array), media_type="application/octet-stream")


@router.get("/heatmap-data", response_model=HeatmapResponse)
async def get_heatmap_data(
        query: str = Query(..., description="Search query to generate heatmap relevance"),
        limit: int = Query(2000, description="Max points to return")
):
    """
    专门为 3D 视图设计的高性能接口。
    不返回完整元数据，只返回坐标和相关性分数。
    """
    # 限制最大值防止前端崩掉
    safe_limit = min(limit, 5000)

    points = search_service.get_heatmap_data(query, safe_limit)

    return HeatmapResponse(
        status="success",
        count=len(points),
        data=points
    )
