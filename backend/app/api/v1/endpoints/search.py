from fastapi import APIRouter, HTTPException, Depends
from backend.app.schema.search import TextSearchRequest, SearchResponse
from backend.app.service.search_service import search_service

router = APIRouter()


@router.post("/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    接收前端 JSON -> 调用 Service -> 返回 JSON
    """
    try:
        results = search_service.search_text(
            query=request.query,
            limit=request.limit,
            threshold=request.threshold
        )

        return SearchResponse(
            status="success",
            count=len(results),
            data=results
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
