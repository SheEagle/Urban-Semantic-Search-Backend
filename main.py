import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.utils.global_state import init_resources
from backend.app.api.v1.routers import router as v1_router

app = FastAPI(title="Venice Search System")

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 路由
app.include_router(v1_router, prefix="/api/v1")


# 4. 启动预热 (单例初始化)
@app.on_event("startup")
async def startup_event():
    print("🚀 System Starting... Initializing Global Resources.")
    # 这里会触发 feature_extractor 的加载
    try:
        init_resources()
    except Exception as e:
        print(f"⚠️ Warning: Resource initialization failed: {e}")
        print("Please check if your 'core' folder is in the root directory.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
