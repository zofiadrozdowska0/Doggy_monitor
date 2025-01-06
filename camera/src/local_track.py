import cv2
import socket
import threading
import time
import math
import numpy as np
from flask import Flask, Response

# Import klasy ServoController
from servo_controller import ServoController

class PIDController:
    def __init__(self, K_p, K_i, K_d):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        self.previous_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value, dt, dead_zone=None):
        # Obliczenie błędu
        error = setpoint - measured_value

        # P
        proportional = self.K_p * error

        # I
        self.integral += error * dt
        integral = self.K_i * self.integral

        # D
        derivative = self.K_d * (error - self.previous_error) / dt
        self.previous_error = error

        # Suma PID
        output = proportional + integral + derivative

        # Obsługa strefy martwej (dead zone)
        if dead_zone and abs(output) < dead_zone:
            output = 0
            self.integral = 0  # Reset integral

        return output

    def reset(self):
        """Resetuje stan kontrolera PID."""
        self.previous_error = 0
        self.integral = 0

class Tracker:
    def __init__(self, host='0.0.0.0', port=8487, frame_size=(320, 240)):
        self.host = host
        self.port = port
        self.frame_width, self.frame_height = frame_size
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2

        # Sterowanie serwami
        self.servo_controller = ServoController(servo_pin_x=18, servo_pin_y=4)
        self.current_angle_x, self.current_angle_y = self.servo_controller.read_angles_from_file()

        # PID kontrolery
        self.pid_x = PIDController(K_p=0.02, K_i=0.0002, K_d=0.0035)
        self.pid_y = PIDController(K_p=0.02, K_i=0.0002, K_d=0.0035)

        self.prev_time = time.time()
        self.last_received_time = None  # Czas ostatniego odbioru danych
        self.last_known_position_time = None  # Czas ostatniej znanej pozycji psa
        self.initial_position = (90, 90)  # Pozycja początkowa

        # Filtr Kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # Konfiguracja socketu
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Tracker nasłuchuje na {self.host}:{self.port}")

        # Kamera
        self.picam2 = None

        # Flagi dla działania wątków
        self.running = True
        self.feedback_thread = threading.Thread(target=self.manage_feedback_connections)
        self.feedback_thread.start()
        self.watchdog_thread = threading.Thread(target=self.watchdog)
        self.watchdog_thread.start()

    def start_camera(self):
        # Konfiguracja kamery
        from picamera2 import Picamera2
        from libcamera import Transform

        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"format": 'RGB888', "size": (self.frame_width, self.frame_height)},
            transform=Transform(hflip=False, vflip=True)
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

    def stream_frames(self):
        try:
            while self.running:
                frame = self.picam2.capture_array()  # Pobranie obrazu z kamery

                # Obrót obrazu w poziomie
                flipped_frame = cv2.flip(frame, 1)

                # Kodowanie obrazu do JPEG
                _, buffer = cv2.imencode('.jpg', flipped_frame)

                # Wysłanie danych w formacie MJPEG
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except GeneratorExit:
            print("Przerwano strumieniowanie.")

    def watchdog(self):
        """Monitoruje stan odbioru danych i steruje trybem pracy."""
        search_mode = False

        while self.running:
            current_time = time.time()

            # Liczba detekcji w ciągu ostatnich 2 sekund
            detections_count = len(self.detections_in_last_2_sec) if hasattr(self, 'detections_in_last_2_sec') else 0

            if not search_mode and detections_count >= 7:
                # Włącz tryb szukania psa, jeśli ostatnie 7 detekcji było w ciągu 2 sekund
                search_mode = True
                print("Przejście w tryb szukania psa na podstawie Kalmana.")

            if search_mode and self.last_known_position_time and current_time - self.last_known_position_time <= 3:
                # Kontynuuj śledzenie na podstawie przewidywanych pozycji z Kalmana
                prediction = self.kalman.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])
                print(f"Przewidywana pozycja (po zgubieniu psa): {pred_x}, {pred_y}")
                self.control_servos(pred_x, pred_y)
            elif search_mode and self.last_known_position_time and current_time - self.last_known_position_time > 5:
                # Wyłącz tryb szukania, jeśli minęło 5 sekundy od ostatniej znanej pozycji
                search_mode = False
                print("Zakończenie trybu szukania psa. Reset do pozycji początkowej.")
                self.servo_controller.move(*self.initial_position)
                self.current_angle_x, self.current_angle_y = self.initial_position
                self.pid_x.reset()
                self.pid_y.reset()
                self.last_known_position_time = None
            elif not search_mode and self.last_received_time and current_time - self.last_received_time > 10:
                # Powrót do pozycji początkowej po 10 sekundach braku danych
                print("Brak danych przez 10 sekund, powrót do pozycji początkowej.")
                self.servo_controller.move(*self.initial_position)
                self.current_angle_x, self.current_angle_y = self.initial_position
                self.pid_x.reset()
                self.pid_y.reset()
                self.last_known_position_time = None

            time.sleep(0.1)  # Odświeżanie co 100 ms


    def manage_feedback_connections(self):
        while self.running:
            print("Oczekiwanie na nowe połączenie z serwerem...")
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"Połączono z serwerem: {addr}")

                # Obsługa komunikacji w osobnym wątku
                threading.Thread(target=self.handle_client, args=(client_socket,)).start()
            except Exception as e:
                print(f"Błąd podczas akceptowania połączenia: {e}")
                time.sleep(1)

    def handle_client(self, client_socket):
        buffer = ""
        detections_in_last_2_sec = []  # Przechowuje czas detekcji w ciągu ostatnich 2 sekund
        try:
            while self.running:
                data = client_socket.recv(1024).decode()
                buffer += data
                current_time = time.time()
                dt = current_time - self.prev_time

                if dt <= 0:
                    continue

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        bbox_center_x, bbox_center_y = map(int, line.split(','))
                        self.last_received_time = current_time  # Zaktualizuj czas ostatniego odbioru danych
                        self.last_known_position_time = current_time  # Zaktualizuj czas ostatniej znanej pozycji psa

                        # Dodaj czas detekcji do listy
                        detections_in_last_2_sec.append(current_time)
                        # Usuń przestarzałe detekcje
                        detections_in_last_2_sec = [
                            t for t in detections_in_last_2_sec if current_time - t <= 2
                        ]

                        # Aktualizacja filtra Kalmana na podstawie rzeczywistego pomiaru
                        measured = np.array([[np.float32(bbox_center_x)], [np.float32(bbox_center_y)]])
                        self.kalman.correct(measured)

                        # Przewidywanie kolejnej pozycji
                        prediction = self.kalman.predict()
                        pred_x, pred_y = int(prediction[0]), int(prediction[1])

                        # Wyliczenie następnej pozycji serwa
                        delta_theta_x = self.pid_x.compute(self.frame_center_x, pred_x, dt, dead_zone=1)
                        delta_theta_y = self.pid_y.compute(self.frame_center_y, pred_y, dt, dead_zone=1)

                        new_angle_x = self.current_angle_x + delta_theta_x
                        new_angle_y = self.current_angle_y - delta_theta_y

                        # Ograniczenie kątów
                        new_angle_x = max(0, min(180, new_angle_x))
                        new_angle_y = max(0, min(150, new_angle_y))

                        # Przesuwanie serw
                        self.servo_controller.move(new_angle_x, new_angle_y)

                        # Aktualizacja bieżących kątów
                        self.current_angle_x, self.current_angle_y = new_angle_x, new_angle_y

                        # Zaktualizuj czas
                        self.prev_time = current_time

                    except ValueError:
                        print(f"Błąd w przetwarzaniu linii: {line}")
                        continue

                # Zapisz listę detekcji do obiektu dla `watchdog`
                self.detections_in_last_2_sec = detections_in_last_2_sec

        except socket.error as e:
            print(f"Błąd podczas odbierania danych: {e}")
        finally:
            client_socket.close()
            print("Rozłączono klienta.")


    def control_servos(self, pred_x, pred_y):
        """Steruje serwami na podstawie przewidywanej pozycji."""
        dt = time.time() - self.prev_time
        delta_theta_x = self.pid_x.compute(self.frame_center_x, pred_x, dt, dead_zone=1)
        delta_theta_y = self.pid_y.compute(self.frame_center_y, pred_y, dt, dead_zone=1)

        new_angle_x = self.current_angle_x + delta_theta_x
        new_angle_y = self.current_angle_y - delta_theta_y

        # Ograniczenie kątów
        new_angle_x = max(0, min(180, new_angle_x))
        new_angle_y = max(0, min(150, new_angle_y))

        # Przesuwanie serw
        self.servo_controller.move(new_angle_x, new_angle_y)

        # Aktualizacja bieżących kątów
        self.current_angle_x, self.current_angle_y = new_angle_x, new_angle_y

    def stop(self):
        print("Zamykanie wątków i zasobów...")
        self.running = False

        # Zamknij socket serwera
        if self.server_socket:
            self.server_socket.close()

        # Zatrzymaj kamerę
        if self.picam2:
            self.picam2.stop()

        # Czekaj na zakończenie wątków
        if self.feedback_thread.is_alive():
            self.feedback_thread.join()
        if self.watchdog_thread.is_alive():
            self.watchdog_thread.join()

        print("Zasoby zostały zwolnione.")

# Flask do serwowania wideo
app = Flask(__name__)
tracker = Tracker()

@app.route('/video_feed')
def video_feed():
    return Response(tracker.stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        tracker.start_camera()
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        tracker.stop()
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")
        tracker.stop()
