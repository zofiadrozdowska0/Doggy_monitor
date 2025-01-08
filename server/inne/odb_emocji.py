import socket
import threading

class EmotionReceiver:
    def __init__(self, emotion_port, file_port, output_file):
        self.emotion_port = emotion_port
        self.file_port = file_port
        self.output_file = output_file

        # Gniazdo dla emocji
        self.emotion_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.emotion_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.emotion_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.emotion_socket.bind(("", self.emotion_port))

        # Gniazdo dla pliku
        self.file_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.file_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.file_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.file_socket.bind(("", self.file_port))

    def start_emotion_receiver(self):
        print(f"Nasłuchuję emocji na porcie {self.emotion_port}...")
        try:
            while True:
                data, addr = self.emotion_socket.recvfrom(1024)  # Odbierz dane (do 1024 bajtów)
                message = data.decode()
                print(f"Odebrano emocję od {addr}: {message}")
        except KeyboardInterrupt:
            print("Zatrzymano nasłuchiwanie emocji.")
        finally:
            self.emotion_socket.close()

    def start_file_receiver(self):
        print(f"Nasłuchuję pliku na porcie {self.file_port}...")
        try:
            while True:
                data, addr = self.file_socket.recvfrom(4096)  # Odbierz dane (do 4096 bajtów)
                file_content = data.decode()
                print(f"Odebrano plik od {addr}. Zapisuję jako {self.output_file}.")
                with open(self.output_file, "w") as f:
                    f.write(file_content)
        except KeyboardInterrupt:
            print("Zatrzymano nasłuchiwanie pliku.")
        finally:
            self.file_socket.close()

    def start_receiving(self):
        # Uruchomienie nasłuchiwania w dwóch wątkach
        emotion_thread = threading.Thread(target=self.start_emotion_receiver, daemon=True)
        file_thread = threading.Thread(target=self.start_file_receiver, daemon=True)

        emotion_thread.start()
        file_thread.start()

        # Czekaj na zakończenie wątków
        emotion_thread.join()
        file_thread.join()

if __name__ == "__main__":
    emotion_port = 5005  # Port dla broadcastu emocji
    file_port = 5006     # Port dla broadcastu pliku
    output_file = "odebrany.txt"

    receiver = EmotionReceiver(emotion_port, file_port, output_file)
    receiver.start_receiving()
