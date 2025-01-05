import torch
import cv2
import numpy as np
import pandas as pd

# Załaduj model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ustawienia analizy
frame_skip = 3

# Otwórz plik wideo
video_path = '15.mp4'  # Wprowadź ścieżkę do pliku wideo
cap = cv2.VideoCapture(video_path)

# Sprawdź, czy plik wideo został otwarty prawidłowo
if not cap.isOpened():
    print("Nie można otworzyć pliku wideo")
    exit()

# Parametry zapisu wideo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) / frame_skip  # Dostosowanie FPS do frame_skip
output_path = 'output_with_predictions.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Inicjalizacja filtra Kalmana
kalman = cv2.KalmanFilter(4, 2)  # 4 zmienne stanu, 2 zmienne pomiaru (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

# Inicjalizacja licznika klatek
frame_counter = 0

# Inicjalizacja zmiennych do analizy
distances = []
in_bounding_box_count = 0
total_predictions = 0

# Przechowywanie poprzedniej przewidywanej pozycji
previous_prediction = None
skip_first_frames = 5

# Główna pętla
while cap.isOpened():
    # Przechwyć klatkę z wideo
    ret, frame = cap.read()
    if not ret:
        print("Nie można odczytać klatki lub koniec wideo")
        break

    # Sprawdź, czy analizować bieżącą klatkę
    if frame_counter % frame_skip != 0:
        frame_counter += 1
        continue

    # Konwertuj obraz do RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wykonaj detekcję
    results = model(img_rgb)

    # Przetwórz wyniki
    detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
    dog_class_id = 16  # Klasa dla psa
    dog_detections = [d for d in detections if int(d[5]) == dog_class_id]

    # Przetwarzanie wykryć psa
    if dog_detections:
        # Bierzemy pierwsze wykrycie psa
        dog = dog_detections[0]
        x1, y1, x2, y2 = map(int, dog[:4])
        confidence = dog[4]

        # Oblicz środek prostokąta (x, y)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Aktualizacja filtra Kalmana na podstawie rzeczywistego pomiaru (środek psa)
        measured = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(measured)

        # Przewidywanie kolejnej pozycji przez filtr Kalmana
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        # Zapisz poprzednią przewidywaną pozycję
        if previous_prediction is not None:
            prev_pred_x, prev_pred_y = previous_prediction
            cv2.circle(frame, (prev_pred_x, prev_pred_y), 5, (255, 0, 0), -1)  # Niebieska kropka - poprzednia pozycja

            # Oblicz różnicę w pikselach między rzeczywistą a przewidywaną pozycją
            diff_x = abs(prev_pred_x - center_x)
            diff_y = abs(prev_pred_y - center_y)
            distance = np.sqrt(diff_x**2 + diff_y**2)
            if frame_counter / frame_skip > skip_first_frames:
                distances.append(distance)

                # Sprawdzenie, czy przewidywana pozycja znajduje się wewnątrz bounding boxa
                if x1 <= prev_pred_x <= x2 and y1 <= prev_pred_y <= y2:
                    in_bounding_box_count += 1
                total_predictions += 1

        # Narysowanie rzeczywistego i przewidywanego środka psa
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)  # Rzeczywisty środek - zielony
        cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)  # Przewidywany środek - czerwony
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Dog: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Zapisz tylko analizowane klatki do pliku wideo
        out.write(frame)

        # Wyświetl klatkę
        cv2.imshow('Detected and Predicted Dog Center', frame)

        # Zapisz aktualną przewidywaną pozycję jako poprzednią
        previous_prediction = (pred_x, pred_y)

    # Zwiększ licznik klatek
    frame_counter += 1

    # Przerwij pętlę po naciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby
cap.release()
out.release()
cv2.destroyAllWindows()

# Obliczenie procentu trafień w bounding box
percent_in_bounding_box = (in_bounding_box_count / total_predictions) * 100
print(f"Procent przewidywanych pozycji wewnątrz bounding boxa: {percent_in_bounding_box:.2f}%")

# Wykresy
import matplotlib.pyplot as plt

# Wykres odległości
plt.figure(figsize=(10, 6))
plt.plot(distances, label='Odległość przewidywana-rzeczywista')
plt.xlabel('Klatka')
plt.ylabel('Odległość (piksele)')
plt.title('Odległość przewidywanej pozycji od rzeczywistej')
plt.legend()
plt.grid(True)
plt.savefig('distance_plot.png')  # Zapisanie wykresu odległości
plt.show()

# Histogram błędów przewidywanej pozycji
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Błąd przewidywanej pozycji (piksele)')
plt.ylabel('Liczba wystąpień')
plt.title('Histogram błędów przewidywanej pozycji')
plt.grid(True)
plt.savefig('error_histogram.png')  # Zapisanie histogramu błędów
plt.show()

# Zapisanie distances do CSV
distances_data = pd.DataFrame(distances, columns=["Distance"])
distances_data.to_csv('distances.csv', index_label="Frame")

print("Distances saved to distances.csv")
