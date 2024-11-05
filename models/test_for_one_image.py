import os
import numpy as np
import cv2
from ultralytics import YOLO


model = YOLO('grupa3.pt')
image_path = 'datasets/data/images/test/image_2100.PNG'
img = cv2.imread(image_path)
height, width = img.shape[:2]


results = model(img)


normalized_keypoints = []
if len(results) > 0 and hasattr(results[0], 'keypoints'):
    for keypoint_set in results[0].keypoints.data:
        normalized_points = []
        for x, y, confidence in keypoint_set:
            normalized_x = (x.cpu().item() if x.is_cuda else x.item()) / width
            normalized_y = (y.cpu().item() if y.is_cuda else y.item()) / height
            confidence = confidence.cpu().item() if confidence.is_cuda else confidence.item()
            normalized_points.append((normalized_x, normalized_y, confidence))
        normalized_keypoints.append(normalized_points)


label_path = 'datasets/data/labels/test/image_2100.txt'
labels = []
with open(label_path, 'r') as f:
    for line in f:
        data = line.strip().split()
        class_id = int(data[0])
        bbox_x, bbox_y, bbox_w, bbox_h = map(float, data[1:5])
        keypoints = []
        for i in range(5, len(data), 3):
            px, py, visibility = float(data[i]), float(data[i + 1]), int(data[i + 2])
            keypoints.append((px, py, visibility))
        labels.append((class_id, bbox_x, bbox_y, bbox_w, bbox_h, keypoints))

# Evaluation parameters
threshold = 0.05
confidence_threshold = 0.5
TP = FP = TN = FN = 0

# Initialize lists to track indices of each group
TP_indices = []
FP_indices = []
TN_indices = []
FN_indices = []

for i, label in enumerate(labels):
    class_id, bbox_x, bbox_y, bbox_w, bbox_h, label_keypoints = label


    if len(normalized_keypoints) > i:
        pred_keypoints = normalized_keypoints[i]

        for j, (label_x, label_y, visibility) in enumerate(label_keypoints):

            if j >= len(pred_keypoints):
                continue

            pred_x, pred_y, pred_conf = pred_keypoints[j]

            if visibility == 2:  # Visibility
                if pred_conf >= confidence_threshold:

                    distance = np.sqrt((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2)
                    if distance <= threshold:
                        TP += 1
                        TP_indices.append((i, j))
                    else:
                        FP += 1
                        FP_indices.append((i, j))
                else:
                    FN += 1
                    FN_indices.append((i, j))
            else:
                if pred_conf < confidence_threshold:
                    TN += 1
                    TN_indices.append((i, j))
                else:
                    FP += 1
                    FP_indices.append((i, j))


print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"TN: {TN}")
print(f"FN: {FN}")


TP_indices_only = [j for _, j in TP_indices]
FP_indices_only = [j for _, j in FP_indices]
TN_indices_only = [j for _, j in TN_indices]
FN_indices_only = [j for _, j in FN_indices]


print(f"TP Indices: {TP_indices_only}")
print(f"FP Indices: {FP_indices_only}")
print(f"TN Indices: {TN_indices_only}")
print(f"FN Indices: {FN_indices_only}")


# Calculate percentage metrics
total_predictions = TP + FP + TN + FN
tpr = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
fpr = (FP / (FP + TN)) * 100 if (FP + TN) > 0 else 0
precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
accuracy = ((TP + TN) / total_predictions) * 100 if total_predictions > 0 else 0

# Display percentage metrics
print(f"True Positive Rate (Recall): {tpr:.2f}%")
print(f"False Positive Rate: {fpr:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
