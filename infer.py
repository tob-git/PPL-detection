from ultralytics import YOLO

model = YOLO("models/ppe_yolov8.pt")

results = model.predict(
    source="datasets/construction-ppe/images/test",
    conf=0.25,
    save=True
)

print("Inference done. Check runs/detect/predict/")
