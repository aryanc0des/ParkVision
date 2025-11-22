from ultralytics import YOLO

model = YOLO("yolov8l.pt")

img_path = "data/raw/test.jpg"

results = model(img_path)

results[0].show()