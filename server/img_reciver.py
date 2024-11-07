import csv
import os
import socket
import struct
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Abstrakcyjny interfejs dla źródła obrazu
class ImageSource:
    def get_frame(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def close(self):
        raise NotImplementedError("This method should be overridden by subclasses")


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

        packed_msg_size = self.data[: self.payload_size]
        self.data = self.data[self.payload_size :]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Odbieranie rzeczywistego obrazu
        while len(self.data) < msg_size:
            self.data += self.client_socket.recv(4 * 1024)

        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Dekodowanie obrazu w formacie YUV i konwersja do RGB
        yuv_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (480 * 3 // 2, 640)
        )
        frame_rgb = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB_I420)
        return frame_rgb

    def close(self):
        # Zamknięcie socketu
        self.client_socket.close()


# Implementacja źródła obrazu opartego na pliku wideo (MP4)
class MovieImageSource(ImageSource):
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def close(self):
        self.cap.release()


# Interfejs do renderowania bounding boxa
class BoundingBoxRenderer:
    def render_bbox(
        self,
        frame,
        x1,
        y1,
        x2,
        y2,
        pred_x: int,
        pred_y: int,
        prev_prediction: Tuple[int, int] = None,
        detected=True,
    ):
        raise NotImplementedError("This method should be overridden by subclasses")

    def close(self):
        pass


# Implementacja renderowania bounding boxa na ekranie
class ScreenBoundingBoxRenderer(BoundingBoxRenderer):
    def render_bbox(
        self,
        frame,
        x1,
        y1,
        x2,
        y2,
        pred_x: int,
        pred_y: int,
        prev_prediction: Tuple[int, int] = None,
        detected=True,
    ):
        if detected:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (255, 255, 0), 2
            )  # Bounding box - cyan

            # Draw actual and predicted center points
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2
            # Rysowanie rzeczywistego i przewidywanego środka psa
            cv2.circle(
                frame, (bbox_center_x, bbox_center_y), 5, (0, 255, 0), -1
            )  # Rzeczywisty środek - zielony
            cv2.circle(
                frame, (pred_x, pred_y), 5, (0, 0, 255), -1
            )  # Przewidywany środek - czerwony

            # Jeśli jest dostępna poprzednia przewidywana pozycja, zaznacz ją niebieskim kolorem
            if prev_prediction is not None:
                prev_x, prev_y = prev_prediction
                cv2.circle(
                    frame, (prev_x, prev_y), 5, (255, 0, 0), -1
                )  # Poprzednia przewidywana pozycja - niebieski

        cv2.imshow("Detected and Predicted Dog Center", frame)

        # Wyjście po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        return True

    def close(self):
        cv2.destroyAllWindows()


# Implementacja renderowania bounding boxa i zapisywania do pliku wideo
class VideoBoundingBoxRenderer(BoundingBoxRenderer):
    def __init__(
        self, output_path: str, frame_width: int, frame_height: int, fps: int = 30
    ):
        self.output_path = output_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )

    def render_bbox(
        self,
        frame,
        x1,
        y1,
        x2,
        y2,
        pred_x: int,
        pred_y: int,
        prev_prediction: Tuple[int, int] = None,
        detected=True,
    ):
        if detected:
            # Draw the bounding box
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (255, 255, 0), 2
            )  # Bounding box - cyan

            # Draw actual and predicted center points
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2
            cv2.circle(
                frame, (bbox_center_x, bbox_center_y), 5, (0, 255, 0), -1
            )  # Real center - green
            cv2.circle(
                frame, (pred_x, pred_y), 5, (0, 0, 255), -1
            )  # Predicted center - red

            # Draw previous prediction if available
            if prev_prediction is not None:
                prev_x, prev_y = prev_prediction
                cv2.circle(
                    frame, (prev_x, prev_y), 5, (255, 0, 0), -1
                )  # Previous prediction - blue

        # Write the frame to the video buffer
        self.video_writer.write(frame)

    def close(self):
        self.video_writer.release()


# Implementacja renderowania bounding boxa na ekranie
class NoScreenRenderer(BoundingBoxRenderer):
    def render_bbox(
        self,
        frame,
        x1,
        y1,
        x2,
        y2,
        pred_x: int,
        pred_y: int,
        prev_prediction: Tuple[int, int] = None,
        detected=True,
    ):
        return True

    def close(self):
        pass


# Klasa do wykrywania i śledzenia psa za pomocą Faster R-CNN i filtra Kalmana
class DogTracker:
    def __init__(
        self,
        image_source: ImageSource,
        bbox_renderer: BoundingBoxRenderer,
        bbox_ip,
        bbox_port,
        connect=True,
    ):
        self.image_source = image_source
        self.bbox_renderer = bbox_renderer

        # Ustawienia dla Faster R-CNN i urządzenia (GPU lub CPU)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        # self.model.eval().to(self.device)
        self.net = cv2.dnn.readNetFromCaffe(
            "mobilenet_ssd.prototxt", "mobilenet_ssd.caffemodel"
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CPU
        )  # Set to CPU; use DNN_TARGET_CUDA for CUDA

        # Inicjalizacja filtra Kalmana
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.previous_prediction = (
            None  # Zmienna do przechowywania poprzedniej przewidywanej pozycji
        )

        self.connect = connect
        if connect:
            # Konfiguracja socketu do wysyłania współrzędnych bounding boxa
            self.bbox_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.bbox_socket.connect((bbox_ip, bbox_port))

    def process_frames(self, fun=None):
        frames_pred = 0
        limit = 5
        pred_x = 320
        pred_y = 240
        dog_detected = 0
        while True:
            # Pobieranie klatki obrazu ze źródła
            frame = self.image_source.get_frame()
            if frame is None:
                break

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            dog_idx = None
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    class_id = int(detections[0, 0, i, 1])
                    if class_id == 12:  # Dog class
                        dog_idx = i
                        break
            if dog_idx is not None:
                box = detections[0, 0, dog_idx, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                )
                (x1, y1, x2, y2) = box.astype("int")
                bbox_center_x = x1 + (x2 - x1) // 2
                bbox_center_y = y1 + (y2 - y1) // 2

                frames_pred = 0
                dog_detected = 1

            # if len(dog_idx) > 0:
            #     x1, y1, x2, y2 = boxes[dog_idx[0]].astype(int)
            #     # Obliczanie środka bounding boxa
            #     bbox_center_x = x1 + (x2 - x1) // 2
            #     bbox_center_y = y1 + (y2 - y1) // 2
            #
            #     frames_pred = 0
            #     dog_detected = 1

            elif (
                dog_detected == 1 or (frames_pred < limit and frames_pred > 0)
            ):  # frames_pred ustawiony na przewidywanie max 5 klatek, jeśłi nie widzimy psa
                bbox_center_x = pred_x
                # ignore y movement :)

                frames_pred += 1
                dog_detected = 0

            if dog_detected == 1 or (frames_pred < limit and frames_pred > 0):
                # Aktualizacja filtra Kalmana na podstawie rzeczywistego pomiaru
                measured = np.array(
                    [[np.float32(bbox_center_x)], [np.float32(bbox_center_y)]]
                )
                self.kalman.correct(measured)

                # Przewidywanie kolejnej pozycji przez filtr Kalmana
                prediction = self.kalman.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                # Renderowanie bounding boxa
                self.bbox_renderer.render_bbox(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    pred_x,
                    pred_y,
                    self.previous_prediction,
                )

                # Zaktualizuj poprzednią przewidywaną pozycję na obecną
                self.previous_prediction = (pred_x, pred_y)

                if fun:
                    fun((bbox_center_x, bbox_center_y), (pred_x, pred_y))
                # Wysłanie przewidywanych współrzędnych bounding boxa do Raspberry Pi
                bbox_message = f"{bbox_center_x},{bbox_center_y}\n"
                if self.connect:
                    self.bbox_socket.sendall(bbox_message.encode())

            else:  # resetuj Kalmana jak zgubisz obiekt
                self.kalman = cv2.KalmanFilter(4, 2)
                self.kalman.measurementMatrix = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
                )
                self.kalman.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
                )
                self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
                self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
                self.previous_prediction = None

                self.bbox_renderer.render_bbox(
                    frame, None, None, None, None, None, None, None, False
                )

    def close(self):
        # Zamknięcie źródła obrazu, renderera bounding boxa i socketu na współrzędne bounding boxa
        self.image_source.close()
        if self.connect:
            self.bbox_renderer.close()
            self.bbox_socket.close()


# Klasa do testowania śledzenia
class DogTrackerTest:
    def __init__(self):
        self.differences = []

    def test_tracking(self):
        for video_path in os.listdir("/Users/olaf/Desktop/kotki"):
            print(f"analysing {video_path}")
            # Konfiguracja źródła obrazu z pliku wideo i śledzenia psa
            image_source = MovieImageSource(
                os.path.join("/Users/olaf/Desktop/kotki", video_path)
            )
            # bbox_renderer = VideoBoundingBoxRenderer("output_kot.mp4", 1280, 720, 30)
            bbox_renderer = ScreenBoundingBoxRenderer()
            # bbox_renderer = NoScreenRenderer()  # ScreenBoundingBoxRenderer()
            dog_tracker = DogTracker(
                image_source, bbox_renderer, "192.168.1.248", 8486, False
            )

            def gather_points(bbox, pred):
                diff = np.sqrt((bbox[0] - pred[0]) ** 2 + (bbox[1] - pred[1]) ** 2)
                self.differences = np.append(self.differences, diff)

            try:
                dog_tracker.process_frames(gather_points)
            finally:
                dog_tracker.close()

        self.save_differences()
        self.reduce_differences()
        self.plot_differences()

    def plot_histogram(self):
        # Rysuj histogram różnic
        plt.figure(figsize=(12, 6))
        plt.hist(self.differences, bins=30, edgecolor="black")
        plt.title(
            "Częstotliwość występowania odległości punktu przewidywanego od środka bounding boxa"
        )
        plt.xlabel("Odległość punktu przewidywanego od środka bounding boxa")
        plt.ylabel("Częstotliwość występowania")
        plt.grid(axis="y")
        plt.savefig("lol.png")
        plt.show()

    def load_differences(self):
        # Load differences from the CSV file
        with open("differences.csv", mode="r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header
            self.differences = [float(row[0]) for row in reader]

    def save_differences(self, csv_path="differences.csv"):
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Difference"])  # Header
            for diff in self.differences:
                writer.writerow([diff])

        print(f"Differences saved to {csv_path}")

    def reduce_differences(self, max_values=20):
        # Convert differences to a numpy array
        self.differences = np.array(self.differences)

        # Determine the number of values in each chunk
        chunk_size = max(1, len(self.differences) // max_values)

        # Aggregate by averaging values in chunks
        reduced_differences = [
            np.mean(self.differences[i : i + chunk_size])
            for i in range(0, len(self.differences), chunk_size)
        ]

        # Store the reduced differences back in the instance variable
        self.differences = np.array(
            reduced_differences[:max_values]
        )  # Ensure only max_values are kept

    def normalize_differences(self):
        # Normalizuj różnice, aby były w zakresie 0-1
        self.differences = np.array(self.differences)
        self.differences = (self.differences - self.differences.min()) / (
            self.differences.max() - self.differences.min()
        )

    def plot_differences(self):
        # Rysuj wykres różnic
        plt.figure(figsize=(12, 6))
        plt.plot(self.differences)
        plt.title("Różnice między rzeczywistym a przewidywanym środkiem bounding boxa")
        plt.xlabel("Numer klatki")
        plt.ylabel("Znormalizowana różnica")
        plt.grid()
        plt.savefig("lol.png")


if __name__ == "__main__":
    # Przykład użycia
    image_source = StreamImageSource("192.168.1.248", 8485)
    # bbox_renderer = ScreenBoundingBoxRenderer()
    bbox_renderer = VideoBoundingBoxRenderer("omg.mp4", 640, 480, 10)
    dog_tracker = DogTracker(image_source, bbox_renderer, "192.168.1.248", 8486)

    try:
        dog_tracker.process_frames()
    finally:
        dog_tracker.close()
