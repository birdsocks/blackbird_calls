from ultralytics import YOLO

# Load a model
model = YOLO('/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/YOLO/runs/detect/train14/weights/best.pt')

# Customize validation settings
metrics = model.val(data="blackbird.yaml", imgsz=640, batch=32, conf=0.01, iou=0.7, device="0")
print(metrics.box.map)