import socket
import cv2
import numpy as np
import struct
import torch
from facenet_pytorch import MTCNN1

# Abstrakcyjny interfejs dla źródła obrazu
class ImageSource:
    def get_frame(self):
        raise NotImplementedError("Ta metoda powinna być nadpisana przez klasy dziedziczące")

# Implementacja źródła obrazu opartego na strumieniu socketowym
class StreamImageSource(ImageSource):
    def __init__(self, ip, port):
        # Konfiguracja socketu do odbierania obrazu
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((ip, port))
        self.data = b""
        self.payload_size = struct.calcsize("Q")

    def get_frame(self):
        # Odbieranie nagłówka z rozmiarem danych
        while len(self.data) < self.payload_size:
            packet = self.client_socket.recv(4 * 1024)
            if not packet:
                return None  # Połączenie zamknięte
            self.data += packet

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Odbieranie rzeczywistego obrazu
        while len(self.data) < msg_size:
            self.data += self.client_socket.recv(4 * 1024)
        
        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]
        
        # Dekodowanie obrazu w formacie YUV i konwersja do RGB
        yuv_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((480 * 3 // 2, 640))
        frame_rgb = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
        return frame_rgb

    def close(self):
        # Zamknięcie socketu
        self.client_socket.close()

# Klasa do wykrywania i śledzenia twarzy za pomocą MTCNN i filtra Kalmana
class FaceTracker:
    def __init__(self, image_source, bbox_ip, bbox_port):
        self.image_source = image_source
        # Ustawienia dla MTCNN i urządzenia (GPU lub CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from facenet_pytorch.models.mtcnn import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Inicjalizacja filtra Kalmana
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.previous_prediction = None  # Zmienna do przechowywania poprzedniej przewidywanej pozycji

        # Konfiguracja socketu do wysyłania współrzędnych bounding boxa
        self.bbox_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bbox_socket.connect((bbox_ip, bbox_port))

    def process_frames(self):
        frames_pred = 0
        limit = 5
        pred_x = 320
        pred_y = 240
        before_y = None
        dog_detected = 0
        while True:
            # Pobieranie klatki obrazu ze źródła
            frame = self.image_source.get_frame()
            if frame is None:
                break
            
            # Wykrywanie twarzy za pomocą MTCNN
            boxes, _ = self.mtcnn.detect(frame)

            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                # Rysowanie prostokąta wokół wykrytej twarzy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Obliczanie środka bounding boxa
                bbox_center_x = x1 + (x2 - x1) // 2
                bbox_center_y = y1 + (y2 - y1) // 2

                frames_pred = 0
                dog_detected = 1

            elif dog_detected == 1 or (frames_pred < limit and frames_pred > 0): # frames_pred ustawiony na przewidywanie max 5 klatek, jeśłi nie widzimy psa
                bbox_center_x = pred_x
                # ignore y movement :)


                frames_pred += 1
                dog_detected = 0


            if dog_detected == 1 or (frames_pred < limit and frames_pred > 0):
                # Aktualizacja filtra Kalmana na podstawie rzeczywistego pomiaru
                measured = np.array([[np.float32(bbox_center_x)], [np.float32(bbox_center_y)]])
                self.kalman.correct(measured)

                # Przewidywanie kolejnej pozycji przez filtr Kalmana
                prediction = self.kalman.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                # Rysowanie rzeczywistego i przewidywanego środka twarzy
                cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 255, 0), -1)  # Rzeczywisty środek - zielony
                cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)  # Przewidywany środek - czerwony

                # Jeśli jest dostępna poprzednia przewidywana pozycja, zaznacz ją niebieskim kolorem
                if self.previous_prediction is not None:
                    prev_x, prev_y = self.previous_prediction
                    cv2.circle(frame, (prev_x, prev_y), 5, (255, 0, 0), -1)  # Poprzednia przewidywana pozycja - niebieski

                # Zaktualizuj poprzednią przewidywaną pozycję na obecną
                self.previous_prediction = (pred_x, pred_y)



                # Wysłanie przewidywanych współrzędnych bounding boxa do Raspberry Pi
                bbox_message = f"{bbox_center_x},{bbox_center_y}\n"
                self.bbox_socket.sendall(bbox_message.encode())

            else: #resetuj Kalmana jak zgubisz obiekt
                self.kalman = cv2.KalmanFilter(4, 2)
                self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
                self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
                self.previous_prediction = None

            

            # Wyświetlanie obrazu z wykrytą twarzą
            cv2.imshow("Detected and Predicted Face Center", frame)

            # Wyjście po naciśnięciu 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def close(self):
        # Zamknięcie źródła obrazu i socketu na współrzędne bounding boxa
        self.image_source.close()
        self.bbox_socket.close()
        cv2.destroyAllWindows()

# Przykład użycia
if __name__ == "__main__":
    # Konfiguracja źródła obrazu ze strumienia i śledzenia twarzy
    image_source = StreamImageSource('192.168.137.182', 8485)
    face_tracker = FaceTracker(image_source, '192.168.137.182', 8486)

    try:
        face_tracker.process_frames()
    finally:
        face_tracker.close()
