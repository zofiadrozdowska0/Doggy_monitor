import torch
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('grupa3.pt')

image_path = 'datasets/data/images/test/image_2100.PNG'
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Nie można wczytać obrazu. Sprawdź ścieżkę do pliku.")

results = model(img)

print("Kluczowe punkty dla psa:")

if len(results) > 0 and hasattr(results[0], 'keypoints'):
    for i, keypoint_set in enumerate(results[0].keypoints.data):
        print(f"\nDetekcja {i + 1}:")

        for j, keypoint in enumerate(keypoint_set):
            x, y, confidence = keypoint
            print(f"Punkt {j + 1}: X = {x.item():.2f}, Y = {y.item():.2f}, Confidence = {confidence.item():.2f}")
else:
    print("Nie wykryto punktów kluczowych dla obiektów na obrazie.")

for result in results:
    result.show()
