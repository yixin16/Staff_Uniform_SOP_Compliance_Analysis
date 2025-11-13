import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from ultralytics import YOLO
import json
import numpy as np
import time

# --- Window Capture Imports ---
import ctypes
import win32gui
import win32ui
import win32con

# ==============================================================================
# --- ⚙️ MASTER CONFIGURATION ⚙️ ---
# ==============================================================================

# --- Window Capture Settings ---
WINDOW_TITLE_KEYWORD = "iVMS-4200"
CAMERA_GRID_ROWS = 2
CAMERA_GRID_COLS = 2
ACTIVE_CAMERA_INDEXES = [0, 1] # e.g., [0, 1] for the first two cameras

# --- Model & Inference Settings ---
PERSON_MODEL_PATH = "yolo11n.pt"
SHIRT_MODEL_PATH = "top_wear_detector.pt"
FINETUNED_MODEL_PATH = "best_model_finegrained_swin.pth"
CLASS_MAP_PATH = "class_to_idx.json"
CONFIDENCE_THRESHOLD = 0.80
DROPOUT_RATE = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


COMPLIANT_CAPTURES_DIR = "compliant_captures"
COMPLIANT_LABELS = ["polo_compliant", "vest_compliant"]

# ==============================================================================
# --- END OF CONFIGURATION ---
# ==============================================================================

os.makedirs(COMPLIANT_CAPTURES_DIR, exist_ok=True)

# --- DPI Awareness ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

#region Window Capture Functions
def get_window_by_title(title_keyword, min_width=100, min_height=100):
    def enum_windows(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_keyword.lower() in title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if width >= min_width and height >= min_height:
                    result.append(hwnd)
    results = []
    win32gui.EnumWindows(enum_windows, results)
    return results[0] if results else None

def capture_window(hwnd):
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    width, height = right - left, bot - top
    if width <= 0 or height <= 0: return np.array([]) # Return empty if minimized
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8').reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    return img_bgr

def split_grid_cameras(frame, rows=2, cols=2, active_idxs=None):
    h, w = frame.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    all_cameras = []
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        crop = frame[y1:y2, x1:x2]
        all_cameras.append(crop)
    if active_idxs is None:
        return all_cameras, list(range(len(all_cameras)))
    selected_cameras = [all_cameras[i] for i in active_idxs if i < len(all_cameras)]
    return selected_cameras, active_idxs
#endregion

def main():
    print("--- Headless Real-Time PPE Compliance Monitor (Swin Transformer) ---")
    print(f"Using device: {DEVICE}")
    print(f"Targeting cameras with indexes: {ACTIVE_CAMERA_INDEXES}")
    print(f"Saving compliant captures to: '{COMPLIANT_CAPTURES_DIR}/'")

    # --- 1. Load Models and Mappings ---
    try:
        with open(CLASS_MAP_PATH, 'r') as f: class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print("Loading YOLO models...")
        person_detector = YOLO(PERSON_MODEL_PATH)
        shirt_detector = YOLO(SHIRT_MODEL_PATH)

        # =======================================================================
        # 🔽 --- BLOCK MODIFIED: Load the Swin Transformer model ---
        # =======================================================================
        print("Loading fine-tuned Swin Transformer classifier...")
        # 1. Initialize the Swin-T model architecture without pre-trained weights
        classifier_model = models.swin_t(weights=None)

        # 2. Get the number of input features for the final layer (the 'head')
        num_ftrs = classifier_model.head.in_features

        # 3. Replace the head with a new one matching your trained model's architecture
        classifier_model.head = torch.nn.Sequential(
            torch.nn.Dropout(p=DROPOUT_RATE),
            torch.nn.Linear(num_ftrs, len(idx_to_class))
        )

        # 4. Load your fine-tuned weights
        state_dict = torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
        classifier_model.load_state_dict(state_dict) # Use strict=True (default) is safer

        # 5. Set the model to the correct device and evaluation mode
        classifier_model.to(DEVICE)
        classifier_model.eval()
        # =======================================================================
        # --- END OF MODIFIED BLOCK ---
        # =======================================================================

    except FileNotFoundError as e:
        print(f"❌ Error loading model or file: {e}. Please check your paths.")
        return

    # --- 2. Image Transformations (Unchanged - Swin-T also uses 224x224) ---
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 3. Find Target Window ---
    print(f"Searching for window containing '{WINDOW_TITLE_KEYWORD}'...")
    hwnd = get_window_by_title(WINDOW_TITLE_KEYWORD)
    if not hwnd:
        print(f"❌ FATAL: Window '{WINDOW_TITLE_KEYWORD}' not found.")
        return
    print(f"✅ Window found! Starting real-time analysis...")

    # --- 4. Main Real-Time Inference Loop (Logic remains the same) ---
    try:
        while True:
            if win32gui.IsIconic(hwnd):
                print("Window is minimized. Waiting...", end='\r')
                time.sleep(1)
                continue

            window_frame = capture_window(hwnd)
            if window_frame.size == 0: continue

            camera_feeds, original_indexes = split_grid_cameras(
                window_frame,
                CAMERA_GRID_ROWS,
                CAMERA_GRID_COLS,
                ACTIVE_CAMERA_INDEXES
            )

            for i, cam_frame in enumerate(camera_feeds):
                original_cam_index = original_indexes[i]
                if cam_frame.size == 0: continue

                person_results = person_detector(cam_frame, classes=[0], conf=0.65, verbose=False)
                for result in person_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_crop = cam_frame[y1:y2, x1:x2]
                        if person_crop.size == 0: continue

                        shirt_results = shirt_detector(person_crop, conf=0.75, verbose=False)
                        predicted_label = "Non-compliant"
                        for s_result in shirt_results:
                            for s_box in s_result.boxes:
                                sx1, sy1, sx2, sy2 = map(int, s_box.xyxy[0])
                                shirt_crop = person_crop[sy1:sy2, sx1:sx2]
                                if shirt_crop.size == 0: continue
                                
                                pil_image = Image.fromarray(cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2RGB))
                                image_tensor = inference_transform(pil_image).unsqueeze(0).to(DEVICE)
                                with torch.no_grad():
                                    outputs = classifier_model(image_tensor)
                                    probs = F.softmax(outputs, dim=1)
                                    conf, top_idx = torch.max(probs, 1)
                                    if conf.item() >= CONFIDENCE_THRESHOLD:
                                        predicted_label = idx_to_class[top_idx.item()]
                                # We only need to classify the first detected shirt
                                break
                        
                        # --- Logic to save compliant captures ---
                        if predicted_label in COMPLIANT_LABELS:
                            timestamp = int(time.time())
                            filename = f"capture_cam{original_cam_index + 1}_{predicted_label}_{timestamp}.jpg"
                            save_path = os.path.join(COMPLIANT_CAPTURES_DIR, filename)
                            
                            # Save the crop of the detected person
                            cv2.imwrite(save_path, person_crop)
                            print(f"✅ Compliant person captured! Saved to '{save_path}'", "Confidence:", f"{conf.item():.2f}   ")
                        
                        print(f"Camera {original_cam_index + 1}: Person detected - Status: {predicted_label}")
            
            # A small delay to prevent excessive CPU usage
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n✅ Program stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during the main loop: {e}")
    finally:
        print("Cleaning up resources.")

if __name__ == "__main__":
    main()
