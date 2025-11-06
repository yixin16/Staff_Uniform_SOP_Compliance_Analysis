import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "data_collection"
REFERENCE_DIR = "reference_img"
DEST_DIR = "ppe_dataset"
TRAIN_RATIO = 0.7
AUGMENT_MULTIPLIER = 2
SEED = 42

# --- Utility Functions ---
def create_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_reference_images(reference_dir, target_dir):
    """
    Merge front/back reference folders into unified class folders in data_collection.
    polo_compliant_front + polo_compliant_back → polo_compliant
    """
    print("\n🧩 Merging reference images into data_collection...")

    for ref_class_folder in os.listdir(reference_dir):
        ref_class_path = os.path.join(reference_dir, ref_class_folder)
        if not os.path.isdir(ref_class_path):
            continue

        # Derive unified class name (remove _front/_back)
        base_name = ref_class_folder.replace("_front", "").replace("_back", "")
        target_class_path = os.path.join(target_dir, base_name)
        create_dir(target_class_path)

        for img_name in os.listdir(ref_class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            src_path = os.path.join(ref_class_path, img_name)
            dest_path = os.path.join(target_class_path, img_name)
            shutil.copy(src_path, dest_path)

        print(f"✅ Merged '{ref_class_folder}' → '{base_name}'")

def augment_image(img):
    augmentations = [
        lambda x: x.rotate(random.uniform(-15, 15)),
        lambda x: ImageOps.mirror(x),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.9, 1.3)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
    ]
    return random.choice(augmentations)(img)

def split_dataset(source, dest, train_ratio):
    random.seed(SEED)

    if os.path.exists(dest):
        print(f"\nDestination '{dest}' already exists. Removing it...")
        shutil.rmtree(dest)

    train_dir = os.path.join(dest, 'train')
    val_dir = os.path.join(dest, 'val')
    create_dir(train_dir)
    create_dir(val_dir)

    print("\nSplitting and augmenting dataset...")

    for class_folder in os.listdir(source):
        source_class_path = os.path.join(source, class_folder)
        if not os.path.isdir(source_class_path):
            continue

        train_class_path = os.path.join(train_dir, class_folder)
        val_class_path = os.path.join(val_dir, class_folder)
        create_dir(train_class_path)
        create_dir(val_class_path)

        images = [f for f in os.listdir(source_class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        if len(images) < 10:
            print(f"Class '{class_folder}' has only {len(images)} images. Consider adding more samples.")

        split_index = int(len(images) * train_ratio)
        train_imgs = images[:split_index]
        val_imgs = images[split_index:]

        # Copy + augment training images
        for img_name in tqdm(train_imgs, desc=f"Train [{class_folder}]"):
            src_path = os.path.join(source_class_path, img_name)
            dest_path = os.path.join(train_class_path, img_name)
            try:
                img = Image.open(src_path).convert('RGB')
                img.save(dest_path)

                for i in range(AUGMENT_MULTIPLIER):
                    aug_img = augment_image(img)
                    aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                    aug_img.save(os.path.join(train_class_path, aug_name))
            except Exception as e:
                print(f" Skipped {img_name}: {e}")

        # Copy validation images
        for img_name in tqdm(val_imgs, desc=f"Val [{class_folder}]"):
            src_path = os.path.join(source_class_path, img_name)
            dest_path = os.path.join(val_class_path, img_name)
            try:
                img = Image.open(src_path).convert('RGB')
                img.save(dest_path)
            except Exception as e:
                print(f" Skipped {img_name}: {e}")

        print(f"*** {class_folder}: {len(train_imgs)} train (+{len(train_imgs)*AUGMENT_MULTIPLIER} aug), {len(val_imgs)} val")

    print(f"\n=Dataset ready at: {dest}")

# --- Run ---
if __name__ == "__main__":
    copy_reference_images(REFERENCE_DIR, SOURCE_DIR)
    split_dataset(SOURCE_DIR, DEST_DIR, TRAIN_RATIO)
