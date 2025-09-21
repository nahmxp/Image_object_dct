import os
import cv2
import glob
import random
import albumentations as A
from tqdm import tqdm

# Input dataset paths
IMG_DIR = "./Dataset/YOLO/yolov11/test/images"
LBL_DIR = "./Dataset/YOLO/yolov11/test/labels"

# Output augmented dataset paths
OUT_IMG_DIR = "./Dataset/YOLO_aug/yolov11/test/images"
OUT_LBL_DIR = "./Dataset/YOLO_aug/yolov11/test/labels"
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Albumentations pipeline
transform = A.Compose([
    A.Resize(640, 640),
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
    ], p=0.8),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.ToGray(p=0.3),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# How many new images to create per original
AUG_MULTIPLIER = 5

def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO polygon labels and convert to absolute pixel coords"""
    polygons, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            poly = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                poly.append((x, y))
            polygons.append(poly)
            class_labels.append(cls)
    return polygons, class_labels

def save_yolo_labels(out_path, polygons, class_labels, img_w, img_h):
    with open(out_path, "w") as f:
        for cls, poly in zip(class_labels, polygons):
            norm_poly = []
            for (x, y) in poly:
                nx = x / img_w
                ny = y / img_h
                norm_poly.append((nx, ny))
            flat_poly = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_poly])
            f.write(f"{cls} {flat_poly}\n")

# Process dataset
for img_path in tqdm(glob.glob(os.path.join(IMG_DIR, "*.jpg"))):
    fname = os.path.basename(img_path)
    lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    if img is None or not os.path.exists(lbl_path):
        continue

    h, w = img.shape[:2]

    # Load polygons
    polygons, class_labels = load_yolo_labels(lbl_path, w, h)

    # Flatten polygons into keypoints
    keypoints = []
    poly_lengths = []
    for poly in polygons:
        poly_lengths.append(len(poly))
        keypoints.extend(poly)

    # Generate multiple augmentations
    for i in range(AUG_MULTIPLIER):
        augmented = transform(image=img, keypoints=keypoints)

        aug_img = augmented["image"]
        aug_kps = augmented["keypoints"]

        # Rebuild polygons
        new_polys = []
        idx = 0
        for length in poly_lengths:
            new_polys.append(aug_kps[idx:idx+length])
            idx += length

        # Save with unique name
        out_img_name = fname.replace(".jpg", f"_aug{i}.jpg")
        out_lbl_name = fname.replace(".jpg", f"_aug{i}.txt")

        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), aug_img)
        save_yolo_labels(
            os.path.join(OUT_LBL_DIR, out_lbl_name),
            new_polys, class_labels, aug_img.shape[1], aug_img.shape[0]
        )