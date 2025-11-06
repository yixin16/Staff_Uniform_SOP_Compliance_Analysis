import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from tqdm import tqdm
from torchvision import transforms 


SIMILARITY_THRESHOLD = 0.80
SOURCE_FOLDER = "detected_shirts"
DATA_COLLECTION_DIR = "data_collection"
REFERENCE_IMG_DIR = "reference_img"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_AUGMENTATIONS_PER_IMAGE = 5

print(f"--- Part 2: Parent Class Classification Script (with Augmentation) ---")
print(f"Using device: {DEVICE}")

os.makedirs(DATA_COLLECTION_DIR, exist_ok=True)

print("Loading CLIP model...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
print("Models loaded.")

#  augmentation pipeline ---
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

# --- Prepare Reference Features ---
def get_reference_features(reference_dir):
    ref_features, ref_labels = [], []
    print(f"Processing and augmenting reference images from '{reference_dir}'...")

    if not os.path.exists(reference_dir):
        print(f"Error: Reference directory '{reference_dir}' not found.")
        exit()
    parent_classes = set()
    for category_folder in os.listdir(reference_dir):
        if os.path.isdir(os.path.join(reference_dir, category_folder)):
            if "Non-compliant" in category_folder:
                 parent_classes.add("Non-compliant")
            else:
                parent_class = '_'.join(category_folder.split('_')[:-1])
                if parent_class:
                    parent_classes.add(parent_class)
    for pc in parent_classes:
        os.makedirs(os.path.join(DATA_COLLECTION_DIR, pc), exist_ok=True)
    os.makedirs(os.path.join(DATA_COLLECTION_DIR, "Non-compliant"), exist_ok=True)
    print(f"Data collection directories created: {list(parent_classes)}")

    # --- Feature Extraction Loop ---
    for category_folder in os.listdir(reference_dir):
        category_path = os.path.join(reference_dir, category_folder)
        if os.path.isdir(category_path):
            for filename in tqdm(os.listdir(category_path), desc=f"Augmenting {category_folder}"):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                label = category_folder
                image_path = os.path.join(category_path, filename)
                original_image = Image.open(image_path).convert("RGB")
                
                # 1. Add the ORIGINAL image's features first
                with torch.no_grad():
                    inputs = clip_processor(images=original_image, return_tensors="pt").to(DEVICE)
                    features = clip_model.get_image_features(**inputs)
                    ref_features.append(features)
                    ref_labels.append(label)

                # 2. Create and add AUGMENTED versions
                for _ in range(NUM_AUGMENTATIONS_PER_IMAGE):
                    augmented_image = augmentation_transform(original_image)
                    with torch.no_grad():
                        inputs = clip_processor(images=augmented_image, return_tensors="pt").to(DEVICE)
                        features = clip_model.get_image_features(**inputs)
                        ref_features.append(features)
                        ref_labels.append(label) # They all get the same label

    if not ref_labels:
        print(f"Error: No reference images found. Cannot classify.")
        exit()

    print(f"\nTotal reference features (with augmentations): {len(ref_features)}")
    return torch.cat(ref_features), ref_labels

ref_features, ref_labels = get_reference_features(REFERENCE_IMG_DIR)
print(f"\nStarting classification process for images in '{SOURCE_FOLDER}'...")


for filename in tqdm(os.listdir(SOURCE_FOLDER), desc=f"Classifying {SOURCE_FOLDER}"):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    image_path = os.path.join(SOURCE_FOLDER, filename)
    pil_image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
        live_features = clip_model.get_image_features(**inputs)
    similarities = cosine_similarity(live_features.cpu().numpy(), ref_features.cpu().numpy())
    max_similarity = np.max(similarities)
    best_match_index = np.argmax(similarities)
    predicted_label = ref_labels[best_match_index]
    if max_similarity > SIMILARITY_THRESHOLD:
        if "Non-compliant" in predicted_label:
            save_folder_label = "Non-compliant"
        else:
            save_folder_label = '_'.join(predicted_label.split('_')[:-1])
    else:
        save_folder_label = "Non-compliant"
    if save_folder_label:
        destination_folder = os.path.join(DATA_COLLECTION_DIR, save_folder_label)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(image_path, destination_path)

print(f"\n--- Classification complete! ---")