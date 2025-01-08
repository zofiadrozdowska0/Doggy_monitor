import pigpio
import time
import os
from threading import Thread

class ServoController:
    def __init__(self, servo_pin_x, servo_pin_y, angle_file="servo_angles.txt"):
        self.servo_pin_x = servo_pin_x
        self.servo_pin_y = servo_pin_y
        self.angle_file = angle_file
        self.pi = pigpio.pi()

        # Odczyt początkowych kątów dla serw na osi X i Y
        self.current_angle_x, self.current_angle_y = self.read_angles_from_file()

        # Ustawienie serw na ostatnie pozycje
        self.set_angle(self.servo_pin_x, self.current_angle_x)
        self.set_angle(self.servo_pin_y, self.current_angle_y)
        time.sleep(1)

    # Funkcja ustawiająca kąt
    def set_angle(self, servo_pin, angle):
        pulse_width = 500 + (angle * (2000 / 180))
        self.pi.set_servo_pulsewidth(servo_pin, pulse_width)

    # Funkcja do zmiany kąta dla osi X
    def move_x(self, target_angle_x):
        # Aktualizacja kąta serwa na osi X
        self.set_angle(self.servo_pin_x, target_angle_x)
        self.current_angle_x = target_angle_x

    # Funkcja do zmiany kąta dla osi Y (z ograniczeniem do 150 stopni)
    def move_y(self, target_angle_y):
        # Ograniczenie kąta dla osi Y do maksymalnie 150 stopni
        if target_angle_y > 150:
            target_angle_y = 150

        # Aktualizacja kąta serwa na osi Y
        self.set_angle(self.servo_pin_y, target_angle_y)
        self.current_angle_y = target_angle_y

    # Funkcja do równoczesnego ruchu na osiach X i Y
    def move(self, target_angle_x, target_angle_y):
        # Tworzymy dwa wątki dla równoczesnego ruchu na osiach X i Y
        thread_x = Thread(target=self.move_x, args=(target_angle_x,))
        thread_y = Thread(target=self.move_y, args=(target_angle_y,))

        # Uruchamiamy oba wątki
        thread_x.start()
        thread_y.start()

        # Czekamy na zakończenie obu wątków
        thread_x.join()
        thread_y.join()

        # Zapisanie nowych kątów po zakończeniu ruchów
        self.save_angles_to_file(self.current_angle_x, self.current_angle_y)

    # Funkcja zapisująca kąty do pliku
    def save_angles_to_file(self, angle_x, angle_y):
        with open(self.angle_file, "w") as file:
            file.write(f"{angle_x} {angle_y}")

    # Funkcja odczytująca kąty z pliku
    def read_angles_from_file(self):
        if os.path.exists(self.angle_file):
            with open(self.angle_file, "r") as file:
                try:
                    angles = file.read().split()
                    return float(angles[0]), float(angles[1])
                except (ValueError, IndexError):
                    return 90, 90
        else:
            return 90, 90

    # Funkcja zamykająca połączenie pigpio
    def stop(self):
        self.pi.set_servo_pulsewidth(self.servo_pin_x, 0)
        self.pi.set_servo_pulsewidth(self.servo_pin_y, 0)
        self.pi.stop()
