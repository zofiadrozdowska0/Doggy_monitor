import os
import cv2
import torch
import json
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_duzy_path = 'best_duzy.pt'

model_duzy = YOLO(model_duzy_path).to(device)
print("Model załadowany na:", device)

# punkty
polish_keypoints = [
    "prawa_przednia_lapa", "prawy_przedni_nadgarstek", "prawy_przedni_lokiec",
    "lewa_przednia_lapa", "lewy_przedni_nadgarstek", "lewy_przedni_lokiec",
    "prawa_tylna_lapa", "prawy_tylny_nadgarstek", "prawy_tylny_lokiec",
    "lewa_tylna_lapa", "lewy_tylny_nadgarstek", "lewy_tylny_lokiec",
    "poczatek_ogona", "koniec_ogona", "kark", "gardlo",
    "prawa_podstawa_ucha", "prawy_czubek_ucha", "lewa_podstawa_ucha",
    "lewy_czubek_ucha", "pod_nosem", "dolna_warga", "prawy_kacik_wargi",
    "lewy_kacik_wargi", "czubek_jezyka", "gorne_zeby", "dolne_zeby"
]

# szkielet
polish_skeleton = [
    [25, 23], [12, 13], [3, 16], [23, 22], [8, 9], [17, 18], [19, 24], [13, 14],
    [15, 17], [4, 5], [5, 6], [17, 23], [10, 11], [9, 13], [1, 2], [19, 20],
    [15, 16], [25, 24], [24, 22], [15, 19], [6, 16], [21, 23], [11, 12],
    [2, 3], [13, 15], [24, 21], [7, 8]
]

# mapa z yolo dla cvat
yolo_to_polish_indices = {
    6: 0, 7: 1, 8: 2, 0: 3, 1: 4, 2: 5, 9: 6, 10: 7, 11: 8,
    3: 9, 4: 10, 5: 11, 12: 12, 13: 13, 22: 14, 23: 15,
    15: 16, 19: 17, 14: 18, 18: 19, 16: 20, 17: 21
}

def process_video(video_path, output_json_path):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    annotations = []
    images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model_duzy(frame)

        keypoints_data = [0, 0, 0] * len(polish_keypoints)

        # przepisywanie w odpowiedniej kolejności
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy().reshape(-1, 2)
                if result.keypoints.conf is not None:
                    confs = result.keypoints.conf.cpu().numpy().flatten()

                    for yolo_idx, (kp, conf) in enumerate(zip(keypoints, confs)):
                        if yolo_idx in yolo_to_polish_indices:
                            idx = yolo_to_polish_indices[yolo_idx]
                            x, y = kp
                            visibility = 2 if conf > 0.5 else 0  #  2 (widoczny), 0 (niewidoczny)
                            keypoints_data[idx * 3:idx * 3 + 3] = [int(x), int(y), visibility]

        num_keypoints = keypoints_data[2::3].count(2)

        images.append({
            "id": frame_id + 1,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "file_name": f"frame_{frame_id:06d}.PNG",
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

        # Dodanie danych anotacji do JSON
        annotations.append({
            "id": int(frame_id + 1),
            "image_id": int(frame_id + 1),
            "category_id": 1,
            "segmentation": [],
            "area": float(frame.shape[1] * frame.shape[0]),
            "bbox": [0, 0, int(frame.shape[1]), int(frame.shape[0])],
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "track_id": 0,
                "keyframe": frame_id % 5 == 0  # `keyframe co 5 klatek
            },
            "keypoints": keypoints_data,
            "num_keypoints": num_keypoints
        })

        frame_id += 1

    cap.release()

    # reczna ostatnia klatka bo zawsze jest jedna mniej i cvat chce istnienie conajmniej jednego połączenia jeżeli
    # jest zdefiniowana
    images.append({
        "id": frame_id + 1,
        "width": images[-1]["width"],
        "height": images[-1]["height"],
        "file_name": f"frame_{frame_id:06d}.PNG",
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    })

    annotations.append({
        "id": int(frame_id + 1),
        "image_id": int(frame_id + 1),
        "category_id": 1,
        "segmentation": [],
        "area": float(images[-1]["width"] * images[-1]["height"]),
        "bbox": [0, 0, int(images[-1]["width"]), int(images[-1]["height"])],
        "iscrowd": 0,
        "attributes": {
            "occluded": False,
            "track_id": 0,
            "keyframe": False
        },
        "keypoints": [0, 0, 2] * len(polish_keypoints),
        "num_keypoints": len(polish_keypoints)
    })

    coco_format = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "Dog Keypoints Dataset",
            "url": "",
            "version": "1.0",
            "year": ""
        },
        "categories": [{
            "id": 1,
            "name": "Dog",
            "supercategory": "",
            "keypoints": polish_keypoints,
            "skeleton": polish_skeleton
        }],
        "images": images,
        "annotations": annotations
    }

    # Zapisanie wyników do pliku JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f)

    print(f"Anotacje zapisano do {output_json_path}")

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            output_json_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_labels.json")
            process_video(video_path, output_json_path)


process_folder('filmy')