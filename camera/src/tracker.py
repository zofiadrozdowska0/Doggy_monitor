import socket
from servo_controller import ServoController
import time
import math

class PIDController:
    def __init__(self, K_p, K_i, K_d):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        self.previous_error = 0  # Błąd z poprzedniego kroku
        self.integral = 0  # Skumulowany błąd (dla części całkującej)

    def compute(self, setpoint, measured_value, dt, dead_zone = None):
        # Obliczenie błędu
        error = setpoint - measured_value
        
        # P
        proportional = self.K_p * error
        
        # I
        self.integral += error * dt
        integral = self.K_i * self.integral
        
        # D
        derivative = self.K_d * (error - self.previous_error) / dt
        if abs(derivative) > 0.1:
            derivative = math.copysign(0.1, derivative) 
        if abs(self.integral) > 0.25 / self.K_i:
            self.integral = math.copysign(0.25, integral)

        print(f"d: {derivative}, i: {integral}, e: {error}")
        
        # Zaktualizuj poprzedni błąd
        self.previous_error = error
        
        # Suma PID
        output = proportional + integral + derivative

        if dead_zone:
            if abs(output) < dead_zone:
                output = 0
                self.integral = 0
        
        return output

class Tracker:
    def __init__(self, host='0.0.0.0', port=8486):
        self.host = host
        self.port = port
        self.servo_controller = ServoController(servo_pin_x=18, servo_pin_y=4)

        # Ustawienia obrazu kamery
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2

        # Parametry regulatora PID
        self.pid_x = PIDController(K_p=0.02, K_i=0.0002, K_d=0.0035)
        self.pid_y = PIDController(K_p=0.02, K_i=0.0002, K_d=0.0035)

        self.prev_time = time.time()

        # Konfiguracja socketu
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Controller nasłuchuje na {self.host}:{self.port}")

    def receive_coordinates(self):
        client_socket, addr = self.server_socket.accept()
        print(f"Połączono z: {addr}")
        buffer = ""

        while True:
            # Odbieranie danych
            data = client_socket.recv(1024).decode()
            buffer += data
            current_time = time.time()
            dt = current_time - self.prev_time  # Różnica czasu

            if dt <= 0:
                continue

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    bbox_center_x, bbox_center_y = map(int, line.split(','))
                    # print(f"Otrzymane współrzędne: {bbox_center_x}, {bbox_center_y}")

                    # Wyliczenie następnej pozycji serwa
                    # print("x")
                    delta_theta_x = self.pid_x.compute(self.frame_center_x, bbox_center_x, dt, 3)
                    # print(f"dx: {delta_theta_x}")
                    # print("y")
                    delta_theta_y = self.pid_y.compute(self.frame_center_y, bbox_center_y, dt, 3)
                    # print(f"dy: {delta_theta_y}")

                    # Aktualizacja pozycji serwa na osi X i Y
                    new_angle_x = self.servo_controller.current_angle_x + delta_theta_x
                    new_angle_y = self.servo_controller.current_angle_y - delta_theta_y
                    # if delta_theta_x < 0.5:
                    #     new_angle_x = self.servo_controller.current_angle_x
                    # if delta_theta_y < 0.5:
                    #     new_angle_y = self.servo_controller.current_angle_y

                    # Zabezpieczenie przed wyjściem poza zakres (0-180 dla X, 0-150 dla Y)
                    new_angle_x = max(0, min(180, new_angle_x))
                    new_angle_y = max(0, min(150, new_angle_y))

                    # Przesuwanie serw
                    self.servo_controller.move(new_angle_x, new_angle_y)

                    # Zaktualizuj czas
                    self.prev_time = current_time

                except ValueError:
                    print(f"Błąd w przetwarzaniu linii: {line}")
                    continue

    def stop(self):
        self.server_socket.close()

if __name__ == "__main__":
    tracker = Tracker()
    try:
        tracker.receive_coordinates()
    except KeyboardInterrupt:
        tracker.stop()
