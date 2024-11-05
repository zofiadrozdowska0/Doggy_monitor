import os
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('grupa3_200.pt')

# paths to the test images and labels directories
image_dir = 'datasets/data/images/test'
label_dir = 'datasets/data/labels/test'

# Parameters for evaluation
threshold = 0.05
confidence_threshold = 0.5

TP = FP = TN = FN = 0

for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    label_file = image_file.replace('.PNG', '.txt')
    label_path = os.path.join(label_dir, label_file)

    # Load and process the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    results = model(img)

    # normalizacja
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

    # Evaluate predictions against labels
    for i, label in enumerate(labels):
        class_id, bbox_x, bbox_y, bbox_w, bbox_h, label_keypoints = label

        if len(normalized_keypoints) > i:
            pred_keypoints = normalized_keypoints[i]
        else:
            continue

        for j, (label_x, label_y, visibility) in enumerate(label_keypoints):
            if j >= len(pred_keypoints):
                continue

            pred_x, pred_y, pred_conf = pred_keypoints[j]

            if visibility == 2:  # Visible
                if pred_conf >= confidence_threshold:
                    distance = np.sqrt((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2)
                    if distance <= threshold:
                        TP += 1
                    else:
                        FP += 1
                else:
                    FN += 1
            else:
                if pred_conf < confidence_threshold:
                    TN += 1
                else:
                    FP += 1

# Calculate final metrics
total_predictions = TP + FP + TN + FN
tpr = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
fpr = (FP / (FP + TN)) * 100 if (FP + TN) > 0 else 0
precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
accuracy = ((TP + TN) / total_predictions) * 100 if total_predictions > 0 else 0

print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"True Positive Rate (Recall): {tpr:.2f}%")
print(f"False Positive Rate: {fpr:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
