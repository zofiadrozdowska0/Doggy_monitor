import socket
import struct
from picamera2 import Picamera2
from libcamera import Transform

class Sender:
    def __init__(self, host='0.0.0.0', port=8485, frame_size=(640, 480)):
        self.host = host
        self.port = port
        self.frame_size = frame_size
        self.picam2 = Picamera2()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Sender nasłuchuje na {self.host}:{self.port}")

    def start_camera(self):
        # Konfiguracja kamery do przesyłania surowych danych
        camera_config = self.picam2.create_preview_configuration(
            main={"format": 'YUV420', "size": self.frame_size},
            transform=Transform(hflip=False, vflip=True)
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

    def send_frames(self):
        client_socket, addr = self.server_socket.accept()
        print(f"Połączono z: {addr}")

        while True:
            frame = self.picam2.capture_array("main")  # Pobieranie surowego obrazu

            # Serializacja danych
            data = frame.tobytes()
            message = struct.pack("Q", len(data)) + data
            client_socket.sendall(message)  # Wysłanie danych

    def stop(self):
        self.server_socket.close()
        self.picam2.stop()

if __name__ == "__main__":
    sender = Sender()
    sender.start_camera()
    try:
        sender.send_frames()
    except KeyboardInterrupt:
        sender.stop()
