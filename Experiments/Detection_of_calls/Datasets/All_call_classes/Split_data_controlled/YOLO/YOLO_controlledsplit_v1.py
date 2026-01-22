from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model
train_results = model.train(
    data="/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/YOLO/blackbird.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=32,  # Batch size
    single_cls=True,  # Multi-class training
    patience=60,  # Early stopping patience
    fliplr=0.0,  # Disable horizontal flip augmentation (doesn't make sense for spectrograms)
    flipud=0.0,  # Disable vertical flip augmentation (would flip frequencies)
    scale=0.0,  # Disable scale augmentation (preserves frequency information)
    degrees=0.0,  # Disable rotation augmentation (doesn't make sense for time-frequency data)
    shear=0.0,  # Disable shear augmentation
    hsv_h=0.03,  # Hue augmentation - simulate different color mappings (increased from 0.015)
    hsv_s=0.9,  # Saturation augmentation - helps with different intensity patterns (increased from 0.7)
    hsv_v=0.6,  # Value/brightness augmentation - simulates different amplitude ranges (increased from 0.4)
)