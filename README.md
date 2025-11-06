# 🧥 Staff Uniform SOP Compliance System

An end-to-end computer vision system that automatically detects staff members in CCTV footage and verifies whether they are wearing the correct uniform top.  
The system combines **YOLO-based detection** with a **fine-tuned EfficientNet classifier**, allowing for robust and automated daily compliance scanning.

---

## 🎯 Real-World Use Case

**Objective:** Automatically assess uniform compliance once per day using surveillance cameras (e.g., Camera 05 & Camera 06).

### Workflow Summary
1. Detect **people** in the frame using YOLO.  
2. Detect the **top-wear region** (shirt/vest) from each person using a fine-tuned YOLO trained on a custom dataset.  
3. Classify the top-wear as **Compliant (Polo Compliant / Vest Compliant)** or **Non-compliant** using a fine-tuned **EfficientNet-B0** model.  
4. Generate visual inference results with colored bounding boxes.

---

## 🧩 System Architecture

```
[ CCTV Frame ]
     │
     ▼
 ┌─────────────────────────────┐
 │ YOLOv11n (Person Detector)  │
 └─────────────────────────────┘
     │ Cropped Persons
     ▼
 ┌──────────────────────────────────────────┐
 │ Fine-Tuned YOLOv11n (Top-wear Detector)  │
 └──────────────────────────────────────────┘
     │ Cropped Shirts
     ▼
 ┌─────────────────────────────┐
 │ EfficientNet-B0 Classifier  │
 │ (Compliant / Non-compliant) │
 └─────────────────────────────┘
     │
     ▼
[ Annotated Inference Results ]
```

---

## ⚙️ Project Components

| Component | Description |
|------------|-------------|
| **YOLO (Person Detector)** | Detects staff members in raw CCTV images. |
| **YOLO (Top-wear Detector)** | Detects shirt/vest region for each detected person. |
| **EfficientNet-B0 Classifier** | Classifies each detected shirt as *polo_compliant*, *vest_compliant*, or *non-compliant*. |
| **CLIP (Optional)** | Used for intelligent dataset filtering and visual data sorting during preparation. |

---

## 🧰 Directory Structure

```
.
├── cctv5/                          # CCTV 05 input images
├── cctv6/                          # CCTV 06 input images
│
├── detected_shirts/                # OUTPUT: Cropped shirt regions
├── inference_results/              # OUTPUT: Annotated results
│
├── models/
│   ├── yolo11n.pt                  # Pretrained YOLO model (person)
│   ├── top_wear_detector.pt        # Custom YOLO model (shirt)
│   ├── best_ppe_classifier.pth     # Fine-tuned EfficientNet model
│   └── class_to_idx.json           # Class index map
│
├── scripts/
│   ├── detect_person.py
│   ├── detect_shirt.py
│   ├── classify_detected.py
│   ├── prepare_dataset.py
│   ├── fine_tune_model.py
│   └── main.py                     # Integrated inference pipeline
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 2️⃣ Prepare Dataset
Use the script below to organize and augment your dataset for training.

```bash
python scripts/prepare_dataset.py
```

This will automatically:
- Split data into train/validation sets.
- Apply data augmentation (rotation, flip, brightness, contrast).
- Ensure balanced data across compliant and non-compliant categories.

---

### 3️⃣ Fine-Tune EfficientNet-B0
Run the following command to fine-tune your classification model:
```bash
python scripts/fine_tune_model.py
```

This will:
- Load EfficientNet-B0
- Train on your prepared dataset
- Save model as `best_ppe_classifier.pth`
- Generate `class_to_idx.json` for class mapping

---

### 4️⃣ Run Full Inference Pipeline
Execute:
```bash
python scripts/main.py
```

This script will:
- Load YOLOv11n and EfficientNet models.
- Process all CCTV images from `cctv5/` and `cctv6/`.
- Detect persons and their shirts.
- Classify detected shirts into compliance categories.
- Annotate and save inference results.

---

## 🎨 Visualization Legend

| Label | Meaning | Bounding Box Color |
|--------|----------|--------------------|
| `polo_compliant` | Staff wearing approved polo uniform | 🟡 Yellow |
| `vest_compliant` | Staff wearing approved vest uniform | 🟦 Cyan |
| `non_compliant` | Staff not wearing correct top wear | 🔴 Red |

---

## ⚡ Configuration Overview (`main.py`)

| Variable | Description |
|-----------|-------------|
| `PERSON_MODEL_PATH` | YOLO model for person detection (`yolo11n.pt`) |
| `SHIRT_MODEL_PATH` | YOLO model for shirt detection (`top_wear_detector.pt`) |
| `FINETUNED_MODEL_PATH` | EfficientNet model path |
| `CLASS_MAP_PATH` | Path to class-to-index mapping file |
| `SOURCES_TO_PROCESS` | List of CCTV folders to process |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for classification |
| `COLOR_MAP` | RGB color codes for visualization |

---

## 🧠 Model Summary

| Model | Framework | Purpose | Notes |
|--------|------------|----------|-------|
| YOLOv11n | Ultralytics | Person detection | Lightweight and real-time capable |
| YOLOv11n (custom) | Ultralytics | Top-wear detection | Fine-tuned on staff torso dataset |
| EfficientNet-B0 | PyTorch | Classification | Fine-tuned for compliant vs non-compliant |

---

## 🧪 Example Inference Output

**Example of annotated CCTV frame with compliance results:**

```
[Person Box] → "polo_compliant"
[Person Box] → "Vest_compliant"
[Person Box] → "non_compliant"
```

Each processed image is automatically saved in:
- `inference_results/` → Annotated compliance images  
- `detected_shirts/` → Cropped top-wear regions  

---

## 📅 Deployment Note

This system is designed for **daily uniform compliance scans** (e.g., once per day per camera).  
It can be optionally integrated with a **face recognition attendance system** to correlate:

- Detected faces (identity)
- Uniform compliance status

---
