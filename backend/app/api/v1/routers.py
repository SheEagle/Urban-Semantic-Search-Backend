from fastapi import APIRouter

from .endpoints.search import router as search_router

router = APIRouter()

# 注册 auth 路由
router.include_router(search_router, prefix="/search", tags=["Search"])
