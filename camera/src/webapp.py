from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from camera_stream import Camera

app = FastAPI()
camera = Camera()


@app.get("/")
async def index():
    return Response(
        content='<img src="/stream" width="640" height="480" />', media_type="text/html"
    )


@app.get("/stream")
async def stream():
    return StreamingResponse(
        camera.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.on_event("shutdown")
def shutdown_event():
    camera.stop_recording()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
