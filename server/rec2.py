import cv2
import numpy as np
import requests
import socket

class DogDetectionServer:
    def __init__(self, mjpeg_url, rpi_ip, rpi_port):
        self.mjpeg_url = mjpeg_url
        self.rpi_ip = rpi_ip
        self.rpi_port = rpi_port
        self.stream = None

        # Wczytanie modelu MobileNet SSD
        self.model_path = "MobileNetSSD_deploy.caffemodel"
        self.config_path = "MobileNetSSD_deploy.prototxt"
        self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)

        # Klasa psa w modelu MobileNet SSD (COCO dataset)
        self.DOG_CLASS_ID = 12

    def connect_to_rpi(self):
        # Ustawienie połączenia socket do Raspberry Pi
        self.rpi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rpi_socket.connect((self.rpi_ip, self.rpi_port))
        print(f"Połączono z Raspberry Pi: {self.rpi_ip}:{self.rpi_port}")

    def start_stream(self):
        # Połączenie ze strumieniem MJPEG
        self.stream = requests.get(self.mjpeg_url, stream=True)
        print(f"Połączono z kamerą na {self.mjpeg_url}")

    def process_stream(self):
        bytes_data = b""
        for chunk in self.stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # Początek obrazu JPEG
            b = bytes_data.find(b'\xff\xd9')  # Koniec obrazu JPEG

            if a != -1 and b != -1:
                jpg = bytes_data[a:b + 2]
                bytes_data = bytes_data[b + 2:]

                # Przetwórz obraz JPEG
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

                # Wykrywanie psa
                self.detect_and_send(frame)

                # Wyświetlenie obrazu
                cv2.imshow("Dog Detection", frame)

                # Wyjście po naciśnięciu 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def detect_and_send(self, frame):
        h, w = frame.shape[:2]

        # Przygotowanie obrazu do modelu
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            # Detekcja psa z prawdopodobieństwem > 0.5
            if confidence > 0.5 and class_id == self.DOG_CLASS_ID:
                x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype("int")
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2

                # Rysowanie bounding boxa i rzeczywistego środka
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Zielona kropka - rzeczywisty środek
                cv2.putText(frame, f'Dog: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Wysłanie rzeczywistej pozycji do Raspberry Pi
                bbox_message = f"{center_x},{center_y}\n"
                self.rpi_socket.sendall(bbox_message.encode())
                print(f"Wysłano do Raspberry Pi: {bbox_message}")

    def close(self):
        if self.stream:
            self.stream.close()
        self.rpi_socket.close()
        cv2.destroyAllWindows()

# Przykład użycia
if __name__ == "__main__":
    mjpeg_url = "http://192.168.137.182:5000/video_feed"  # Adres Raspberry Pi
    rpi_ip = "192.168.137.182"  # Adres IP Raspberry Pi
    rpi_port = 8487  # Port, na którym Raspberry Pi odbiera dane

    server = DogDetectionServer(mjpeg_url, rpi_ip, rpi_port)
    try:
        server.connect_to_rpi()
        server.start_stream()
        server.process_stream()
    finally:
        server.close()
