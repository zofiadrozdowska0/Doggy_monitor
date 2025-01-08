import subprocess
import signal
import time

# Uruchomienie sender.py
sender = subprocess.Popen(['python3', 'sender.py'])
print(f"Uruchomiono sender.py z PID: {sender.pid}")

# Uruchomienie tracker.py
tracker = subprocess.Popen(['python3', 'tracker.py'])
print(f"Uruchomiono tracker.py z PID: {tracker.pid}")

def stop_processes():
    print("Zatrzymywanie programów...")

    # Zatrzymanie programów po naciśnięciu Ctrl+C
    sender.terminate()  # Wysyłanie sygnału terminate
    tracker.terminate()

    # Czekanie, aż programy zakończą działanie
    sender.wait()
    tracker.wait()

    print(f"Zakończono sender.py z PID: {sender.pid}")
    print(f"Zakończono tracker.py z PID: {tracker.pid}")

try:
    # Program główny działa i czeka na zakończenie procesów
    while True:
        time.sleep(1)  # Monitorowanie co 1 sekundę

except KeyboardInterrupt:
    # Zatrzymywanie procesów po przerwaniu (Ctrl+C)
    stop_processes()

# Dodatkowe zatrzymanie procesów na wypadek, gdyby któryś z nich nie zakończył się poprawnie
finally:
    if sender.poll() is None:  # Sprawdzenie, czy sender wciąż działa
        sender.terminate()  # Wysyłanie sygnału terminate
        sender.wait()
    if tracker.poll() is None:  # Sprawdzenie, czy tracker wciąż działa
        tracker.terminate()  # Wysyłanie sygnału terminate
        tracker.wait()

print("Program zakończony.")
