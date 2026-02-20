import os
import random
import json
import numpy as np
import trimesh
import nibabel as nib
import thingi10k

from scipy.ndimage import gaussian_filter
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count


# ---------------- CONFIG ----------------
NNUNET_RAW = "./Datasets"
DATASET_ID = 1
DATASET_NAME = "CT_Scans"

VOL_SIZE = 256
GAUSS_BLUR = (0.6, 0.6, 0.6)
NOISE_STD = 0.005
NUM_CASES = 5000

MATERIALS = {
    "plastic": 0.1,
    "aluminum": 0.2,
    "titanium": 0.5,
    "steel": 0.7,
    "copper": 0.8,
    "ceramic": 0.3
}


# ---------------- VOXEL WRITE (FIXED) ----------------
def voxelize_into_volume(mesh, vol, label, mat_id, mu):
    voxel = mesh.voxelized(pitch=1.2)
    pts = voxel.points

    # Weltkoordinaten -> Volumenindex
    idx = np.floor(pts + VOL_SIZE // 2).astype(int)

    if idx.ndim != 2 or idx.shape[0] == 0:
        print(f"Warnung: Mesh {mesh} erzeugt keine Voxels!")
        return  # Mesh überspringen

    valid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < VOL_SIZE) &
        (idx[:, 1] >= 0) & (idx[:, 1] < VOL_SIZE) &
        (idx[:, 2] >= 0) & (idx[:, 2] < VOL_SIZE)
    )

    if np.sum(valid) == 0:
        print(f"Warnung: Mesh {mesh} liegt komplett außerhalb des Volumens!")
        return  # Mesh überspringen

    z, y, x = idx[valid].T

    # spätere Objekte überschreiben frühere
    vol[z, y, x] = mu
    label[z, y, x] = mat_id

# ---------------- SAMPLE GENERATION ----------------
def generate_sample():
    for attempt in range(5):  # max 5 Versuche
        vol = np.zeros((VOL_SIZE, VOL_SIZE, VOL_SIZE), dtype=np.float32)
        label = np.zeros_like(vol, dtype=np.uint16)

        entries = list(thingi10k.dataset(closed=True, manifold=True, solid=True, self_intersecting=False))
        num_objects = random.randint(2, 5)
        picks = random.sample(entries, num_objects)
        cluster_center = np.random.uniform(-30, 30, 3)

        mat_id = 1
        used_materials = []

        for entry in picks:
            vertices, facets = thingi10k.load_file(entry['file_path'])
            mesh = trimesh.Trimesh(vertices, facets)
            mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
            scale_factor = random.uniform(0.8, 1.2)
            mesh.apply_scale(scale_factor)
            offset = cluster_center + np.random.normal(0, 20, 3)
            mesh.apply_translation(offset)
            mat_name = random.choice(list(MATERIALS.keys()))
            mu = MATERIALS[mat_name]
            used_materials.append(mat_name)

            voxelize_into_volume(mesh, vol, label, mat_id, mu)
            mat_id += 1

        if vol.max() > 0:  # Es wurde mindestens ein Voxel gesetzt
            # CT Artefakte
            vol = gaussian_filter(vol, sigma=GAUSS_BLUR)
            vol += np.random.normal(scale=NOISE_STD, size=vol.shape).astype(np.float32)
            z = np.arange(VOL_SIZE)
            vol += 0.001 * np.sin(2 * np.pi * z / 20)[:, None, None]

            meta = {
                "num_objects": num_objects,
                "materials_used": used_materials,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return vol, label, meta
        else:
            print("Warnung: Alle Meshes erzeugten keine Voxels, neuer Versuch...")

    # Falls nach 5 Versuchen noch leer
    raise RuntimeError("Keine Voxels erzeugt nach 5 Versuchen!")

# ---------------- WORKER ----------------
def process_case(i):
    case_id = f"case_{i:04d}"
    try:
        print(f"[PID {os.getpid()}] Start {case_id}")
        vol, lbl, meta = generate_sample()

        if vol.max() == 0:
            print(f"[PID {os.getpid()}] Warnung {case_id}: Case ist komplett leer!")

        img_path = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(labelsTr, f"{case_id}.nii.gz")

        nib.save(nib.Nifti1Image(vol, affine), img_path)
        nib.save(nib.Nifti1Image(lbl, affine), lbl_path)

        print(f"[PID {os.getpid()}] Done {case_id}")
        return case_id, meta, int(lbl.max())

    except Exception as e:
        print(f"[PID {os.getpid()}] Error {case_id}: {e}")
        return case_id, None, 0

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import thingi10k
    thingi10k.init()  # Einmal im Main-Prozess

    dataset_folder = os.path.join(
        NNUNET_RAW, f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    )
    imagesTr = os.path.join(dataset_folder, "imagesTr")
    labelsTr = os.path.join(dataset_folder, "labelsTr")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    affine = np.eye(4)

    num_workers = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())))
    print(f"Using {num_workers} CPU cores")
    with Pool(num_workers) as pool:
        results = pool.map(process_case, range(NUM_CASES))

    # dataset.json
    all_meta = {}
    max_label_id = 0
    for cid, meta, lblmax in results:
        all_meta[cid] = meta
        max_label_id = max(max_label_id, lblmax)

    labels_dict = {"background": 0}
    for k in range(1, max_label_id + 1):
        labels_dict[f"part_{k:02d}"] = k

    dataset_json = {
        "name": DATASET_NAME,
        "tensorImageSize": "3D",
        "channel_names": {"0": "CT"},
        "labels": labels_dict,
        "numTraining": NUM_CASES,
        "file_ending": ".nii.gz",
    }

    with open(os.path.join(dataset_folder, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    with open(os.path.join(dataset_folder, "meta_generated.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    print("Done.")
