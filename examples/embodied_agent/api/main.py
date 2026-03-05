from fastapi import FastAPI

from .routes.debug_frame import router as debug_frame_router
from .routes.health import router as health_router
from .routes.video_stream import router as video_stream_router

app = FastAPI()
app.include_router(debug_frame_router)
app.include_router(health_router)
app.include_router(video_stream_router)


@app.get("/")
async def root():
    return {"message": "ok"}
