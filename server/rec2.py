import socket
import threading
import cv2
import numpy as np
import requests
from process_stream import ProcessStream
from datetime import datetime

class DogDetectionServer:
    def __init__(self, mjpeg_url, rpi_ip, rpi_port, file_path):
        self.mjpeg_url = mjpeg_url
        self.rpi_ip = rpi_ip
        self.rpi_port = rpi_port
        self.file_path = file_path  # Ścieżka do pliku tekstowego
        self.stream = None

        # TCP serwer do transmisji emocji
        self.emotion_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.emotion_server_socket.bind(("0.0.0.0", 5005))  # Wiązanie z dowolnym adresem na porcie 5005
        self.emotion_server_socket.listen(5)  # Maksymalna liczba oczekujących połączeń

        # Wczytanie modelu MobileNet SSD
        # self.model_path = "MobileNetSSD_deploy.caffemodel"
        # self.config_path = "MobileNetSSD_deploy.prototxt"
        # self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)

        # Klasa psa w modelu MobileNet SSD (COCO dataset)
        # self.DOG_CLASS_ID = 12


        # Wczytanie modelu do detekcji psa
        self.dog_detect = ProcessStream(model_path="model_1.pt")
        self.ostatnia_emocja = None

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

        # h, w = frame.shape[:2]

        # # Przygotowanie obrazu do modelu
        # blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        # self.net.setInput(blob)
        # detections = self.net.forward()

        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]
        #     class_id = int(detections[0, 0, i, 1])

        #     # Detekcja psa z prawdopodobieństwem > 0.5
        #     if confidence > 0.5 and class_id == self.DOG_CLASS_ID:
        #         x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype("int")
        #         center_x = x1 + (x2 - x1) // 2
        #         center_y = y1 + (y2 - y1) // 2

        #         # Rysowanie bounding boxa i rzeczywistego środka
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Zielona kropka - rzeczywisty środek
        #         cv2.putText(frame, f'Dog: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #         # Wysłanie rzeczywistej pozycji do Raspberry Pi
        #         bbox_message = f"{center_x},{center_y}\n"
        #         self.rpi_socket.sendall(bbox_message.encode())
        #         print(f"Wysłano do Raspberry Pi: {bbox_message}")

        frame, pred_boxes, emotion = self.dog_detect.process_frame(frame)
        
        if pred_boxes is not None:
            xmin, ymin, xmax, ymax = pred_boxes[0]
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # Wysłanie rzeczywistej pozycji do Raspberry Pi
            bbox_message = f"{center_x},{center_y}\n"
            self.rpi_socket.sendall(bbox_message.encode())

            # Zapis emocji asynchronicznie
            self.zapisz_emocje_async(emotion)

    def handle_emotion_client(self, conn, addr):
        try:
            print(f"Połączono z klientem: {addr}")
            data = conn.recv(1024).decode().strip()
            if data.lower() == "start_emotion":
                print(f"Otrzymano start_emotion od: {addr}")
                while True:
                    if self.ostatnia_emocja:
                        conn.sendall(f"{self.ostatnia_emocja}\n".encode())
                    else:
                        # Jeśli brak nowej emocji, wysyłaj "brak danych" co 1 sekundę
                        conn.sendall("brak danych\n".encode())
                    threading.Event().wait(1)  # Czekaj 1 sekundę
        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"Klient {addr} rozłączył się: {e}")
        except Exception as e:
            print(f"Błąd w połączeniu z klientem {addr}: {e}")
        finally:
            conn.close()


    def start_emotion_server(self):
        print("Serwer emocji uruchomiony, oczekiwanie na połączenia...")
        while True:
            conn, addr = self.emotion_server_socket.accept()
            threading.Thread(target=self.handle_emotion_client, args=(conn, addr), daemon=True).start()

    def zapisz_emocje(self, emotion):
        if emotion != self.ostatnia_emocja:  # Sprawdzaj, czy emocja się zmieniła
            czas = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            with open("emocje.txt", "a") as f:
                f.write(f"{czas} - {emotion}\n")
            self.ostatnia_emocja = emotion  # Aktualizuj ostatnią emocję

    def zapisz_emocje_async(self, emotion):
        threading.Thread(target=self.zapisz_emocje, args=(emotion,)).start()

    def close(self):
        if self.stream:
            self.stream.close()
        self.rpi_socket.close()
        self.emotion_server_socket.close()
        cv2.destroyAllWindows()

# Przykład użycia
if __name__ == "__main__":
    mjpeg_url = "http://192.168.137.182:5000/video_feed"
    rpi_ip = "192.168.137.182"
    rpi_port = 8487
    file_path = "emocje.txt"  # Ścieżka do pliku tekstowego

    server = DogDetectionServer(mjpeg_url, rpi_ip, rpi_port, file_path)
    try:
        server.connect_to_rpi()
        server.start_stream()

        # Uruchomienie serwera emocji w osobnym wątku
        threading.Thread(target=server.start_emotion_server, daemon=True).start()

        server.process_stream()
    finally:
        server.close()
