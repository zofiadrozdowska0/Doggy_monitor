from picamera2 import Picamera2
from picamera2.outputs import FileOutput
from picamera2.encoders import JpegEncoder
import io
from threading import Condition
import numpy as np


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class Camera:
    def __init__(self):
        # Initialize the camera and output
        output = StreamingOutput()
        picam2 = Picamera2()
        picam2.configure(
            picam2.create_preview_configuration(
                buffer_count=1, main={"format": "RGB888", "size": (640, 480)}
            )
        )
        encoder = JpegEncoder()
        picam2.start_recording(encoder, FileOutput(output))

    async def generate_frames(self):
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def stop(self):
        picam2.stop_recording()
