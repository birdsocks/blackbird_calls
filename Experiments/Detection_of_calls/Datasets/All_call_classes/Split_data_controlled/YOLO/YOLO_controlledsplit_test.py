from ultralytics import YOLO
import os

# Path to your trained model
model_path = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/YOLO/runs/detect/train6/weights/best.pt"

# Load the trained model
model = YOLO(model_path)

# Path to dataset configuration
data_yaml = "/home/Shelby/blackbird_calls/Experiments/Detection_of_calls/Datasets/All_call_classes/Split_data_controlled/YOLO/blackbird.yaml"

# Run validation/testing on the test set
results = model.val(
    data=data_yaml,
    split='test',  # Use test split instead of val
    save_json=True,  # Save results to JSON
    save_hybrid=False,  # Save label+prediction hybrid results
    conf=0.25,  # Confidence threshold (adjust as needed)
    iou=0.7,  # IoU threshold for NMS
    plots=True,  # Generate plots
    verbose=True  # Print detailed results
)

# Print test set metrics
print("\n" + "=" * 70)
print("TEST SET EVALUATION RESULTS")
print("=" * 70)
print(f"\nPrecision:    {results.box.p.mean():.4f}")
print(f"Recall:       {results.box.r.mean():.4f}")
print(f"mAP@0.5:      {results.box.map50:.4f}")
print(f"mAP@0.5-0.95: {results.box.map:.4f}")

# Calculate F1 score
precision = results.box.p.mean()
recall = results.box.r.mean()
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"F1 Score:     {f1:.4f}")

print("\n" + "=" * 70)
print("Results saved to: runs/detect/val/")
print("=" * 70)