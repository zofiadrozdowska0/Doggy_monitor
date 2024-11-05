from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n-pose.pt")

    model.train(data='config.yaml', epochs=10, imgsz=640, device=0)
