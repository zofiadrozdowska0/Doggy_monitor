import socket

class EmotionClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None

    def connect_to_server(self):
        # Inicjalizacja i połączenie z serwerem
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.server_port))
        print(f"Połączono z serwerem {self.server_ip}:{self.server_port}")

    def send_start_emotion(self):
        # Wysłanie wiadomości `start_emotion` do serwera
        try:
            self.client_socket.sendall("start_emotion\n".encode())
            print("Wysłano: start_emotion")
        except Exception as e:
            print(f"Błąd podczas wysyłania wiadomości: {e}")
            self.close_connection()

    def receive_emotions(self):
        # Odbieranie emocji od serwera
        try:
            while True:
                data = self.client_socket.recv(1024).decode().strip()
                if not data:
                    print("Serwer zakończył połączenie.")
                    break
                print(f"Otrzymano emocję: {data}")
        except (ConnectionResetError, BrokenPipeError):
            print("Serwer zamknął połączenie.")
        except Exception as e:
            print(f"Błąd podczas odbierania danych: {e}")
        finally:
            self.close_connection()


    def close_connection(self):
        # Zamknięcie połączenia
        if self.client_socket:
            self.client_socket.close()
            print("Zamknięto połączenie z serwerem.")

# Przykład użycia
if __name__ == "__main__":
    server_ip = "localhost"  # IP serwera
    server_port = 5005            # Port serwera (zgodny z kodem serwera)

    client = EmotionClient(server_ip, server_port)

    try:
        client.connect_to_server()
        client.send_start_emotion()
        client.receive_emotions()
    finally:
        client.close_connection()
