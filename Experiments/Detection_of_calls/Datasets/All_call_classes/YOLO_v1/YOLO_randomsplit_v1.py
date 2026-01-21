from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model
train_results = model.train(
    data="/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/YOLO_v1/blackbird.yaml",  # Path to dataset configuration file
    epochs=200,  # Number of training epochs
    imgsz=1280,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=8,  # Batch size
)