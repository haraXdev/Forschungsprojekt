import os
from glob import glob
import json
import nibabel as nib
import numpy as np
import napari
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATASET_FOLDER = r"C:/uniDev/fProject/open/images"
IMAGES_DIR = os.path.join(DATASET_FOLDER, "image")
LABELS_DIR = os.path.join(DATASET_FOLDER, "label")

# ---------------- OPTIONAL: LOAD LABEL NAMES ----------------
dataset_json_path = os.path.join(DATASET_FOLDER, "dataset.json")
label_names = {}

if os.path.exists(dataset_json_path):
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
    label_names = dataset_json.get("labels", {})
    print("Loaded label names:", label_names)

# ---------------- FIND IMAGE–LABEL PAIRS ----------------
image_files = sorted(glob(os.path.join(IMAGES_DIR, "*.nii*")))

assert len(image_files) > 0, "No image files found!"

# Pick one example (change index if needed)
img_path = image_files[0]

# Assume same filename in label folder
base_name = os.path.basename(img_path)
lbl_path = os.path.join(LABELS_DIR, base_name)

assert os.path.exists(lbl_path), f"Label not found for {base_name}"

print(f"Image: {img_path}")
print(f"Label: {lbl_path}")

# ---------------- LOAD NIFTI FILES ----------------
img_nii = nib.load(img_path)
lbl_nii = nib.load(lbl_path)

img = img_nii.get_fdata().astype(np.float32)
lbl = lbl_nii.get_fdata().astype(np.uint16)

plt.imshow(lbl[128], vmin = 0, vmax= 5)
plt.colorbar()
plt.show()
print("Image shape:", img.shape)
print("Label shape:", lbl.shape)
print("Unique labels:", np.unique(lbl))

assert img.shape == lbl.shape, "Image and label shape mismatch!"

# ---------------- NAPARI VIEWER ----------------
viewer = napari.Viewer(ndisplay=3)

# CT image
viewer.add_image(
    img,
    name="CT",
    colormap="gray",
    contrast_limits=(np.percentile(img, 1), np.percentile(img, 99)),
)

# Label mask
viewer.add_labels(
    lbl,
    name="Segmentation",
    opacity=0.5
)

napari.run()