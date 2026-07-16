#!/usr/bin/env python3
"""YOLOv8 Training Script for Red and Green Box Detection"""

from ultralytics import YOLO
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the nano model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data=os.path.join(script_dir, 'WRO detection.v1i.yolov8', 'data.yaml'),
    epochs=1,
    imgsz=320,
    batch=16,
    patience=20,
    device='cpu',  # CPU device
    verbose=True,
    save=True,  # Ensure weights are saved
    project=os.path.join(script_dir, 'runs'),
    name='detect/train',
)

# Print results
print("Training completed!")
print(f"Results: {results}")
