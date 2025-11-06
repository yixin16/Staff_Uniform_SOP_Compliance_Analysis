import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# --- Configuration ---
DETECTED_PERSONS_DIR = "detected_persons"
DETECTED_SHIRTS_DIR = "detected_shirts"
ANNOTATED_DIR = "annotated_shirts"
YOLO_MODEL_PATH = "top_wear_detector.pt"
CONFIDENCE_THRESHOLD = 0.85
SAVE_THRESHOLD = 0.90  # Only save shirts >= 0.90 confidence

print(f"--- Shirt Detection Stage ---")

os.makedirs(DETECTED_SHIRTS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
print(f"Shirt crops will be saved to: '{DETECTED_SHIRTS_DIR}'")
print(f"Annotated results will be saved to: '{ANNOTATED_DIR}'")

# --- Load model ---
print("Loading YOLO shirt detector model...")
shirt_model = YOLO(YOLO_MODEL_PATH)
print("Model loaded successfully.")

# --- Process Each Detected Person Image ---
print("\nStarting shirt detection process...")
for filename in os.listdir(DETECTED_PERSONS_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    person_path = os.path.join(DETECTED_PERSONS_DIR, filename)
    frame = cv2.imread(person_path)
    if frame is None:
        print(f"Warning: Could not read image '{filename}'. Skipping.")
        continue

    # Run YOLO inference for shirt/top-wear
    results = shirt_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    for result in results:
        # Draw bounding boxes on the annotated copy
        annotated_frame = result.plot()

        # Save annotated image
        annotated_path = os.path.join(ANNOTATED_DIR, f"annotated_{filename}")
        cv2.imwrite(annotated_path, annotated_frame)

        # Extract each detected shirt region
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Only save shirts with confidence ≥ 0.90
            if conf < SAVE_THRESHOLD:
                continue

            cropped_shirt = frame[y1:y2, x1:x2]
            if cropped_shirt.size == 0:
                continue

            # Save the cropped shirt image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            shirt_filename = f"{os.path.splitext(filename)[0]}_{timestamp}_conf{conf:.2f}.png"
            save_path = os.path.join(DETECTED_SHIRTS_DIR, shirt_filename)
            cv2.imwrite(save_path, cropped_shirt)

            print(f"Shirt detected (conf={conf:.2f}) and saved: {save_path}")

print("\n--- Shirt detection complete! ---")
print(f"Review cropped shirts in '{DETECTED_SHIRTS_DIR}' and annotated images in '{ANNOTATED_DIR}'.")
