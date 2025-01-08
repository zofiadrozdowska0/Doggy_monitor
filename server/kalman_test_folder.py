import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Załaduj model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Ustawienia analizy
frame_skip = 1

def analyze_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku wideo: {video_path}")
        return None, 0, 0

    # Inicjalizacja filtra Kalmana
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    # Analiza klatek
    distances = []
    in_bounding_box_count = 0
    total_predictions = 0
    previous_prediction = None
    frame_counter = 0
    skip_first_frames = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % frame_skip != 0:
            frame_counter += 1
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.xyxy[0]
        dog_class_id = 16
        dog_detections = [d for d in detections if int(d[5]) == dog_class_id]

        if dog_detections:
            dog = dog_detections[0]
            x1, y1, x2, y2 = map(int, dog[:4])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            measured = np.array([[np.float32(center_x)], [np.float32(center_y)]])
            kalman.correct(measured)
            prediction = kalman.predict()
            pred_x, pred_y = int(prediction[0]), int(prediction[1])

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

            previous_prediction = (pred_x, pred_y)

        frame_counter += 1

    cap.release()
    # out.release()

    if total_predictions > 0:
        percent_in_bounding_box = (in_bounding_box_count / total_predictions) * 100
    else:
        percent_in_bounding_box = 0

    print(f"{video_path}: {percent_in_bounding_box:.2f}% trafień w bounding box")

    return distances, in_bounding_box_count, total_predictions


def analyze_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_distances = []
    total_in_bounding_box = 0
    total_predictions = 0
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Analizuję wideo: {video_path}")
        distances, in_bounding_box_count, predictions = analyze_video(video_path, output_folder)
        if distances:
            all_distances.extend(distances)
            total_in_bounding_box += in_bounding_box_count
            total_predictions += predictions

    if total_predictions > 0:
        overall_percent_in_bounding_box = (total_in_bounding_box / total_predictions) * 100
    else:
        overall_percent_in_bounding_box = 0

    print(f"Łączny procent trafień w bounding box: {overall_percent_in_bounding_box:.2f}%")

    # Histogram błędów dla wszystkich filmów
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Błąd przewidywanej pozycji (piksele)')
    plt.ylabel('Liczba wystąpień')
    plt.title('Histogram błędów przewidywanej pozycji - Wszystkie filmy')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'all_videos_error_histogram.png'))
    plt.close()

    # Zapisanie distances do CSV
    distances_data = pd.DataFrame(all_distances, columns=["Distance"])
    distances_data.to_csv(os.path.join(output_folder, 'distances.csv'), index_label="Frame")


# Ścieżki do folderów
input_folder = 'filmy'  # Zmień na folder z filmami
output_folder = 'output_folder'  # Zmień na folder do zapisu wyników

# Analiza folderu
analyze_folder(input_folder, output_folder)

