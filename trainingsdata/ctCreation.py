import os
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import napari
from datetime import datetime, timezone
import json
import nibabel as nib

# ---------------- CONFIG ----------------
CADS_DIR = "./cads"
NNUNET_RAW = "./nnUNet_raw"
DATASET_ID = 1
DATASET_NAME = "CADSynthetic"  # -> Dataset001_CADSynthetic

VOL_SIZE = 256
GAUSS_BLUR = (0.6, 0.6, 0.6)
NOISE_STD = 0.005

MATERIALS = {
    "air": 0.0001,
    "plastic": 0.1,
    "aluminum": 0.2,
    "titanium": 0.5,
    "steel": 0.7,
    "copper": 0.8,
    "ceramic": 0.3
}

# ---------------- UTILS ----------------
def make_grid(shape):
    z = np.arange(shape[0]) - (shape[0] - 1) / 2
    y = np.arange(shape[1]) - (shape[1] - 1) / 2
    x = np.arange(shape[2]) - (shape[2] - 1) / 2
    return np.meshgrid(z, y, x, indexing="ij")


def add_ellipsoid(label, vol, center, radii, mat_id, mu, alpha=1.0):
    Z, Y, X = make_grid(label.shape)
    cz, cy, cx = center
    rz, ry, rx = radii
    eq = ((Z - cz) / rz) ** 2 + ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
    mask = eq <= 1.0
    vol[mask] = alpha * mu + (1 - alpha) * vol[mask]
    label[mask] = mat_id


def add_block(label, vol, center, size, mat_id, mu, alpha=1.0):
    Z, Y, X = make_grid(label.shape)
    cz, cy, cx = center
    sz, sy, sx = size
    mask = (
        (np.abs(Z - cz) <= sz / 2)
        & (np.abs(Y - cy) <= sy / 2)
        & (np.abs(X - cx) <= sx / 2)
    )
    vol[mask] = alpha * mu + (1 - alpha) * vol[mask]
    label[mask] = mat_id


def add_cylinder(label, vol, center, radius, height, axis, mat_id, mu, alpha=1.0):
    Z, Y, X = make_grid(label.shape)
    cz, cy, cx = center
    if axis == "z":
        mask = ((X - cx) ** 2 + (Y - cy) ** 2 <= radius**2) & (np.abs(Z - cz) <= height / 2)
    elif axis == "y":
        mask = ((X - cx) ** 2 + (Z - cz) ** 2 <= radius**2) & (np.abs(Y - cy) <= height / 2)
    else:  # axis=="x"
        # NOTE: keeping your logic style; this branch isn't used in your current pipeline anyway
        mask = ((Y - cy) ** 2 + (Z - cz) ** 2 <= radius**2) & (np.abs(X - cz) <= height / 2)
    vol[mask] = alpha * mu + (1 - alpha) * vol[mask]
    label[mask] = mat_id


def add_sphere(label, vol, center, radius, mat_id, mu, alpha=1.0):
    add_ellipsoid(label, vol, center, (radius, radius, radius), mat_id, mu, alpha)


# ---------------- GENERATE SAMPLE ----------------
def generate_sample(mesh_file):
    vol = np.zeros((VOL_SIZE, VOL_SIZE, VOL_SIZE), dtype=np.float32)
    label = np.zeros_like(vol, dtype=np.uint16)

    mesh = trimesh.load(mesh_file)
    submeshes = mesh.split()  # separate Teile erkennen

    used_materials = []
    mat_id_counter = 1

    # ---------------- Sub-Meshes voxelisieren ----------------
    for sub in submeshes:
        voxel = sub.voxelized(pitch=1.0, method="subdivide")
        shape = voxel.matrix.shape
        start = [(VOL_SIZE - s) // 2 for s in shape]
        slices = tuple(slice(st, st + s) for st, s in zip(start, shape))

        vol[slices] = voxel.matrix.astype(np.float32) * MATERIALS["aluminum"]
        label[slices][voxel.matrix] = mat_id_counter

        used_materials.append("aluminum")
        mat_id_counter += 1

    # # ---------------- Zufällige zusätzliche Formen ----------------
    # extra_count = np.random.randint(2,5)
    # for i in range(extra_count):
    #     mats = [m for m in MATERIALS.keys() if m not in used_materials and m!="air"]
    #     if not mats: break
    #     mat = np.random.choice(mats)
    #     used_materials.append(mat)
    #     mu = MATERIALS[mat]
    #     mat_id = mat_id_counter
    #     mat_id_counter += 1

    #     shape_type = np.random.choice(["ellipsoid","block","sphere","cylinder"])
    #     center = tuple(np.random.uniform(-60,60,3))
    #     alpha = np.random.uniform(0.3,1.0)

    #     if shape_type=="ellipsoid":
    #         radii = tuple(np.random.uniform(10,40,3))
    #         add_ellipsoid(label, vol, center, radii, mat_id, mu, alpha)
    #     elif shape_type=="block":
    #         size = tuple(np.random.uniform(10,50,3))
    #         add_block(label, vol, center, size, mat_id, mu, alpha)
    #     elif shape_type=="sphere":
    #         radius = np.random.uniform(5,30)
    #         add_sphere(label, vol, center, radius, mat_id, mu, alpha)
    #     elif shape_type=="cylinder":
    #         radius = np.random.uniform(5,25)
    #         height = np.random.uniform(10,50)
    #         axis = np.random.choice(["x","y","z"])
    #         add_cylinder(label, vol, center, radius, height, axis, mat_id, mu, alpha)


    # ---------------- Gaussian Blur + Rauschen ----------------
    vol = gaussian_filter(vol, sigma=GAUSS_BLUR)
    vol += np.random.normal(scale=NOISE_STD, size=vol.shape).astype(np.float32)

    # ---------------- Sinusstreifen-Artefakte ----------------
    z = np.arange(VOL_SIZE)
    vol += 0.001 * np.sin(2 * np.pi * z / 20)[:, None, None]

    meta = {
        "mesh_file": os.path.basename(mesh_file),
        "materials_used": used_materials,
        "material_mu": {m: float(MATERIALS[m]) for m in used_materials},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return vol, label, meta


# ---------------- MAIN ----------------
if __name__ == "__main__":
    dataset_folder = os.path.join(NNUNET_RAW, f"Dataset{DATASET_ID:03d}_{DATASET_NAME}")
    imagesTr = os.path.join(dataset_folder, "imagesTr")
    labelsTr = os.path.join(dataset_folder, "labelsTr")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    # CADs einsammeln
    exts = (".stl", ".obj", ".ply", ".glb", ".gltf")
    cad_files = sorted(
        os.path.join(CADS_DIR, f)
        for f in os.listdir(CADS_DIR)
        if f.lower().endswith(exts)
    )

    if not cad_files:
        raise RuntimeError(f"No CAD files found in '{CADS_DIR}' with extensions {exts}")

    affine = np.eye(4)  # 1mm isotrope Voxel, keine Rotation

    all_meta = {}
    max_label_id = 0

    for i, mesh_path in enumerate(cad_files):
        case_id = f"case_{i:04d}"

        vol, lbl, meta = generate_sample(mesh_path)

        # track labels for dataset.json
        max_label_id = max(max_label_id, int(lbl.max()))
        all_meta[case_id] = meta

        # nnU-Net naming conventions
        img_path = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(labelsTr, f"{case_id}.nii.gz")

        vol_nii = nib.Nifti1Image(vol.astype(np.float32), affine)
        lbl_nii = nib.Nifti1Image(lbl.astype(np.uint16), affine)

        vol_nii.header.set_xyzt_units("mm")
        lbl_nii.header.set_xyzt_units("mm")

        nib.save(vol_nii, img_path)
        nib.save(lbl_nii, lbl_path)

        print(f"[OK] {os.path.basename(mesh_path)} -> {case_id}")

    # dataset.json
    # You currently generate per-submesh labels (1..N). nnU-Net wants a label map.
    labels_dict = {"background": 0}
    for k in range(1, max_label_id + 1):
        labels_dict[f"part_{k:02d}"] = k

    dataset_json = {
        "name": DATASET_NAME,
        "description": "Synthetic CT-like volumes generated from CAD meshes (per-submesh labels)",
        "tensorImageSize": "3D",
        "channel_names": {"0": "CT"},
        "labels": labels_dict,
        "numTraining": len(cad_files),
        "numTest": 0,
        "file_ending": ".nii.gz",
    }

    with open(os.path.join(dataset_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # optional: save meta for traceability
    with open(os.path.join(dataset_folder, "meta_generated.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    print("\nDone!")
    print(f"Dataset written to: {dataset_folder}")

    # Optional: visualize the last generated sample in Napari
    # (keeps your original visualization behavior)
    # viewer = napari.Viewer(ndisplay=3)
    # viewer.add_image(vol, name="volume", colormap="gray")
    # viewer.add_labels(lbl, name="labels")
    # napari.run()
