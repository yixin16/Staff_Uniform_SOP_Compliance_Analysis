import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from ultralytics import YOLO
import json

# --- Configuration ---
PERSON_MODEL_PATH = "yolo11n.pt"
SHIRT_MODEL_PATH = "top_wear_detector.pt"
FINETUNED_MODEL_PATH = "best_ppe_classifier.pth"
CLASS_MAP_PATH = "class_to_idx.json"
SOURCES_TO_PROCESS = ["cctv5", "cctv6"]
OUTPUT_DIR = "inference_results"
CROPPED_DIR = "detected_shirts"
CONFIDENCE_THRESHOLD = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


COLOR_MAP = {
    "polo_compliant": (0, 255, 255),   # Yellow
    "vest_compliant": (255, 255, 0),   # Cyan
    "Non-compliant": (0, 0, 255)       # Red
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

print("--- Person Detection + Shirt Classification (Draw Person Box Only) ---")
print(f"Using device: {DEVICE}")

# --- 1. Load class mapping ---
with open(CLASS_MAP_PATH, 'r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# --- 2. Load models ---
print("Loading YOLO models...")
person_detector = YOLO(PERSON_MODEL_PATH)
shirt_detector = YOLO(SHIRT_MODEL_PATH)

print("Loading fine-tuned EfficientNet classifier...")
classifier_model = models.efficientnet_b0(weights=None)
num_ftrs = classifier_model.classifier[1].in_features
classifier_model.classifier[1] = torch.nn.Linear(num_ftrs, len(idx_to_class))
classifier_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
classifier_model.to(DEVICE)
classifier_model.eval()

# --- 3. Transformations ---
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. Process each CCTV folder ---
for source_folder in SOURCES_TO_PROCESS:
    if not os.path.exists(source_folder):
        print(f"Folder '{source_folder}' not found, skipping...")
        continue

    print(f"\n--- Processing folder: {source_folder} ---")
    for filename in os.listdir(source_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(source_folder, filename)
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        print(f"  - Processing {filename}...", end='')
        detected = False

        # Step 1: Detect persons
        person_results = person_detector(frame, classes=[0], conf=0.65, verbose=False)

        for result in person_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # Step 2: Detect shirt inside this person region
                shirt_results = shirt_detector(person_crop, conf=0.8, verbose=False)
                predicted_label = "Non-compliant"
                confidence = 0.0

                for s_result in shirt_results:
                    for s_box in s_result.boxes:
                        sx1, sy1, sx2, sy2 = map(int, s_box.xyxy[0])
                        shirt_crop = person_crop[sy1:sy2, sx1:sx2]
                        if shirt_crop.size == 0:
                            continue

                        # Step 3: Classify shirt
                        pil_image = Image.fromarray(cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2RGB))
                        image_tensor = inference_transform(pil_image).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            outputs = classifier_model(image_tensor)
                            probs = F.softmax(outputs, dim=1)
                            conf, top_idx = torch.max(probs, 1)
                            conf = conf.item()
                            pred_class = idx_to_class[top_idx.item()]

                        if conf >= CONFIDENCE_THRESHOLD:
                            predicted_label = pred_class
                            confidence = conf
                        else:
                            predicted_label = "Non-compliant"
                            confidence = conf

                        # Save cropped shirt for verification
                        cropped_save_path = os.path.join(
                            CROPPED_DIR, f"{os.path.splitext(filename)[0]}_shirt.jpg"
                        )
                        cv2.imwrite(cropped_save_path, shirt_crop)

                # Step 4: Draw only the person bounding box
                color = COLOR_MAP.get(predicted_label, (0, 0, 0))
                label_text = f"{predicted_label}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                detected = True

        if detected:
            save_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(save_path, frame)
            print(f"Saved to {OUTPUT_DIR}")
        else:
            print("No detections found.")

print("\n✅ Inference complete.")
print(f"Processed results in '{OUTPUT_DIR}' and cropped shirts in '{CROPPED_DIR}'.")
