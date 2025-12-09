from fastapi import APIRouter, HTTPException, Depends
from backend.app.schema.search import TextSearchRequest, SearchResponse
from backend.app.service.search_service import search_service
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

router = APIRouter()


@router.post("/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    接收前端 JSON -> 调用 Service -> 返回 JSON
    """

    results = search_service.search_text(
        query=request.query,
        limit=request.limit,
        threshold=request.threshold,
        filters=request.filters  # 传入 filter
    )

    return SearchResponse(
        status="success",
        count=len(results),
        data=results
    )

# print(f"❌ Error: {e}")
# raise HTTPException(status_code=500, detail=str(e))


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
        file: UploadFile = File(...),
        limit: int = Form(10),  # 从 Form Data 获取参数
        threshold: float = Form(0.0)
):
    """
    接收前端上传的图片文件 -> 调用 Service -> 返回结果
    """
    try:
        # 读取图片二进制内容
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
