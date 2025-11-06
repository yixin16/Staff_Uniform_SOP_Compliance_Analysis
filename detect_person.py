import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# --- Configuration ---
SOURCES_TO_PROCESS = ["cctv5", 'cctv6']
DETECTED_PERSONS_DIR = "detected_persons"
YOLO_MODEL_PATH = "yolo11n.pt"

print(f"--- Person Detection and Cropping Script ---")

# --- Setup Directories ---
os.makedirs(DETECTED_PERSONS_DIR, exist_ok=True)
print(f"Detected persons will be saved to: '{DETECTED_PERSONS_DIR}'")

# --- Load YOLO Model ---
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("Model loaded.")

# --- Main Processing Loop ---
print("\nStarting detection process...")
for source_folder in SOURCES_TO_PROCESS:
    if not os.path.exists(source_folder):
        print(f"Warning: Source folder '{source_folder}' not found. Skipping.")
        continue
    print(f"--- Processing source: {source_folder} ---")

    for filename in os.listdir(source_folder):
        # Process only common image file types
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(source_folder, filename)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image '{filename}'. Skipping.")
            continue

        # Run YOLO detection. We only care about class '0' which is 'person'.
        results = yolo_model(frame, classes=[0], conf=0.65, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_person = frame[y1:y2, x1:x2]
                if cropped_person.size == 0:
                    continue

                # --- Save the Cropped Image ---
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crop_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.png"
                save_path = os.path.join(DETECTED_PERSONS_DIR, crop_filename)

                # Save the image to the disk
                cv2.imwrite(save_path, cropped_person)
                print(f"  - Detected and saved a person to '{save_path}'")

print("\n--- Detection and cropping complete! ---")
print(f"Please review the images in the '{DETECTED_PERSONS_DIR}' folder.")