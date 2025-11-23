from ultralytics import YOLO

#model = YOLO("yolov8l.pt") #OLDER MODEL (NOT FINE TUNED)#

model = YOLO("model/weights/best.pt") #NEWER FINE TUNED MODEL#

img_path = "data/raw/test.jpg"

results = model(img_path)

results[0].show()