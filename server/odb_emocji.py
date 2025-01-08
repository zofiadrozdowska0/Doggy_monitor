import socket

class EmotionReceiver:
    def __init__(self, port):
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.socket.bind(("", self.port))  # Nasłuchuj na wszystkich interfejsach

    def start_receiving(self):
        print(f"Nasłuchuję danych broadcast na porcie {self.port}...")
        try:
            while True:
                data, addr = self.socket.recvfrom(1024)  # Odbierz dane (do 1024 bajtów)
                message = data.decode()
                print(f"Odebrano wiadomość od {addr}: {message}")
        except KeyboardInterrupt:
            print("Zatrzymano nasłuchiwanie.")
        finally:
            self.socket.close()

if __name__ == "__main__":
    broadcast_port = 5005  # Ten sam port, na którym serwer wysyła broadcasty
    receiver = EmotionReceiver(broadcast_port)
    receiver.start_receiving()
