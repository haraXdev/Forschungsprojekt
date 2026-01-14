"""
trainer_object_boundary_unet_watershed.py

Goal (Option 3 workable variant):
- Train a standard 3D UNet from CT intensities to predict:
  (1) object mask (FG vs BG)
  (2) boundary map between parts
- At inference: run watershed inside the object to obtain a labeled parts mask (1..K).
  Labels are arbitrary per volume (no global meaning), but separate parts.

Input channels: 1  (CT)
Output channels: 2 (object_prob, boundary_prob) with sigmoid

Data layout (nnU-Net raw):
  <data_root>/
    imagesTr/
      case_0000_0000.nii.gz
      ...
    labelsTr/
      case_0000.nii.gz   (0=background, 1..N parts within this sample)
"""

import os, glob, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    EnsureTyped, RandFlipd, RandRotate90d, RandCropByPosNegLabeld,
    ScaleIntensityd
)
from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

# For watershed postprocess (CPU)
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from monai.transforms import CenterSpatialCropd
from monai.transforms import CropForegroundd
from monai.transforms import CropForegroundd, SpatialPadd, CenterSpatialCropd


# -------------------------
# 1) Dataset listing
# -------------------------
def make_pairs(images_dir, labels_dir):
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    if len(imgs) == 0:
        raise RuntimeError(f"No NIfTI files found in {images_dir}")

    data = []
    for img in imgs:
        base = os.path.basename(img)

        # nnU-Net often uses labelsTr/case_XXXX.nii.gz while imagesTr has _0000
        # Try exact match first; if not found, also try stripping "_0000"
        lbl1 = os.path.join(labels_dir, base)
        lbl2 = os.path.join(labels_dir, base.replace("_0000", ""))

        if os.path.exists(lbl1):
            lbl = lbl1
        elif os.path.exists(lbl2):
            lbl = lbl2
        else:
            raise FileNotFoundError(f"Label missing for {img}: tried {lbl1} and {lbl2}")

        data.append({"image": img, "label": lbl})
    return data

def weighted_bce(logits, targets, eps=1e-6):
    pos = targets.sum()
    neg = targets.numel() - pos
    pos_weight = (neg / (pos + eps)).clamp(1.0, 100.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

# -------------------------
# 2) Boundary target creation
# -------------------------
def make_object_and_boundary_targets(label_1dhw: torch.Tensor):
    """
    label_1dhw: (1, D, H, W) int, 0=bg, 1..N=parts

    Returns:
      obj: (1, D, H, W) float {0,1}
      bnd: (1, D, H, W) float {0,1}  boundary voxels where neighbor has different label
    """
    lbl = label_1dhw[0].long()  # (D,H,W)
    obj = (lbl > 0).float().unsqueeze(0)

    # boundary = voxel that has at least one 6-neighbor with different label (inside object)
    # We'll compute differences along each axis and mark boundaries on both sides.
    bnd = torch.zeros_like(lbl, dtype=torch.bool)

    # z neighbors
    dz = lbl[1:, :, :] != lbl[:-1, :, :]
    bnd[1:, :, :] |= dz
    bnd[:-1, :, :] |= dz

    # y neighbors
    dy = lbl[:, 1:, :] != lbl[:, :-1, :]
    bnd[:, 1:, :] |= dy
    bnd[:, :-1, :] |= dy

    # x neighbors
    dx = lbl[:, :, 1:] != lbl[:, :, :-1]
    bnd[:, :, 1:] |= dx
    bnd[:, :, :-1] |= dx

    # keep boundaries only where object exists (optional but usually helpful)
    bnd = (bnd & (lbl > 0))

    return obj, bnd.float().unsqueeze(0)


# -------------------------
# 3) Watershed post-process
# -------------------------
def parts_from_object_and_boundary(obj_prob: np.ndarray, bnd_prob: np.ndarray,
                                  obj_thr=0.5, bnd_thr=0.3, min_size=200):
    """
    obj_prob: (D,H,W) float
    bnd_prob: (D,H,W) float

    Returns:
      parts: (D,H,W) int, 0=bg, 1..K parts
    """
    obj = obj_prob > obj_thr
    if obj.sum() == 0:
        return np.zeros_like(obj_prob, dtype=np.int32)

    # Treat low-boundary regions as "inside-part" areas; boundaries act like walls.
    # Compute distance transform inside object but penalize boundaries.
    # One simple trick: erode object by boundary zones to create seeds.
    bnd = bnd_prob > bnd_thr
    interior = obj & (~bnd)

    # Connected components of interior become markers (seeds)
    markers, n = ndi.label(interior)
    if n == 0:
        # fallback: use object as a single part
        out = obj.astype(np.int32)
        out[obj] = 1
        return out

    # Elevation map: boundaries should be high to prevent merging.
    # Use boundary probability as elevation; add small term to prefer compact splits.
    elevation = bnd_prob.astype(np.float32)

    # Watershed inside object mask
    parts = watershed(elevation, markers=markers, mask=obj)

    # Remove tiny regions
    if min_size > 0:
        out = np.zeros_like(parts, dtype=np.int32)
        cur = 0
        for rid in np.unique(parts):
            if rid == 0:
                continue
            m = parts == rid
            if m.sum() >= min_size:
                cur += 1
                out[m] = cur
        parts = out

    return parts.astype(np.int32)


# -------------------------
# 4) Main training
# -------------------------
def train(
    data_root,
    patch_size=(128, 128, 128),
    batch_size=1,
    epochs=50,
    lr=1e-4,
    device="cuda",
    cache_rate=0.2,
):
    files = make_pairs(os.path.join(data_root, "imagesTr"), os.path.join(data_root, "labelsTr"))
    random.shuffle(files)
    if len(files) < 2:
        train_split = files
        val_files = files
    else:
        n_val = max(1, int(0.2 * len(files)))
        #val_files = files[:n_val]
        #train_split = files[n_val:] if len(files[n_val:]) > 0 else files
        val_files = files
        train_split = files


    print(f"Found total: {len(files)} | train: {len(train_split)} | val: {len(val_files)}")

    tf = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),

    CropForegroundd(keys=["image", "label"], source_key="label"),   # crop around label
    SpatialPadd(keys=["image", "label"], spatial_size=patch_size),  # ensure >= patch
    CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
])


    train_ds = CacheDataset(train_split, transform=tf, cache_rate=cache_rate, num_workers=0)
    val_ds   = CacheDataset(val_files,  transform=tf, cache_rate=cache_rate, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=list_data_collate)

    # UNet predicts: channel0=object logit, channel1=boundary logit
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Losses:
    # - object: Dice + BCE
    # - boundary: BCE (Dice for boundaries is often unstable because boundaries are thin)
    dice_obj = DiceLoss(sigmoid=True)
    bce = torch.nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")  # for object only
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            img = batch["image"].to(device)  # (B,1,D,H,W)
            lbl = batch["label"].to(device)  # (B,1,D,H,W) int parts

            # build targets per batch item
            obj_t, bnd_t = [], []
            for b in range(img.shape[0]):
                o, bd = make_object_and_boundary_targets(lbl[b])
                obj_t.append(o)
                bnd_t.append(bd)
            obj_t = torch.stack(obj_t, dim=0).to(device)  # (B,1,D,H,W)
            bnd_t = torch.stack(bnd_t, dim=0).to(device)  # (B,1,D,H,W)

            opt.zero_grad(set_to_none=True)

            logits = model(img)  # (B,2,D,H,W)
            obj_logit = logits[:, 0:1]
            bnd_logit = logits[:, 1:2]

            if epoch == 1 and steps == 0:
                print("img shape:", tuple(img.shape), "lbl shape:", tuple(lbl.shape))
                print("fg_ratio:", (obj_t.sum() / obj_t.numel()).item())

                p = torch.sigmoid(obj_logit)
                print("prob mean/min/max:",
                p.mean().item(),
                p.min().item(),
                p.max().item())

            loss_obj = dice_obj(obj_logit, obj_t) + weighted_bce(obj_logit, obj_t)
            loss_bnd = weighted_bce(bnd_logit, bnd_t)
            loss = loss_obj + 0.5 * loss_bnd  # weight boundary a bit lower

            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().item())
            steps += 1

        epoch_loss /= max(1, steps)

        # -------- VAL (object dice) --------
        # -------- VAL (hard dice) --------
        model.eval()
        hard_dices = []

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)

                obj_t = (lbl > 0).float()                 # (B,1,D,H,W)
                obj_prob = torch.sigmoid(model(img)[:,0:1])
                obj_pred = (obj_prob > 0.5).float()

                inter = (obj_pred * obj_t).sum(dim=(1,2,3,4))
                denom = obj_pred.sum(dim=(1,2,3,4)) + obj_t.sum(dim=(1,2,3,4))
                dice = (2 * inter / (denom + 1e-8))

                hard_dices.append(dice.mean().item())

        mean_dice = float(np.mean(hard_dices))
        print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | val_obj_dice={mean_dice:.4f}")


# -------------------------
# 5) Example inference usage
# -------------------------
@torch.no_grad()
def infer_parts(model, ct_tensor_1dhw: torch.Tensor, device="cuda"):
    """
    ct_tensor_1dhw: (1,D,H,W) torch float
    Returns:
      parts_mask: (D,H,W) int32, 0=bg, 1..K parts
    """
    model.eval()
    x = ct_tensor_1dhw.unsqueeze(0).to(device)  # (1,1,D,H,W)
    logits = model(x)  # (1,2,D,H,W)
    obj_prob = torch.sigmoid(logits[:, 0:1])[0, 0].detach().cpu().numpy()
    bnd_prob = torch.sigmoid(logits[:, 1:2])[0, 0].detach().cpu().numpy()
    return parts_from_object_and_boundary(obj_prob, bnd_prob)


if __name__ == "__main__":
    train(
        data_root=r"C:/uniDev/forschungsprojekt/trainingsdata/nnUNet_raw/Dataset001_CADSynthetic",
        patch_size=(64, 64, 64),
        batch_size=1,
        epochs=300,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_rate=0.2,
    )
