"""
trainer_interactive_labeled_clicks_unet.py

Interactive trainer (as discussed):

Input:
  - CT volume (1 channel)
  - Prompt mask where the user "clicks" voxels and assigns LABEL IDs (1..K)
    * Labels have NO global meaning across volumes.
    * They are just IDs for "the region I clicked as label i" in THIS volume.

How we make this trainable:
  - We fix a maximum number of interactive labels: KMAX (default 8).
  - We convert the integer prompt mask into KMAX one-hot channels.
  - So model input channels = 1 + KMAX.

Output:
  - Multi-class segmentation with (KMAX + 1) channels:
      0 = background
      1..KMAX = prompted regions
  - Note: during each training step we only prompt a subset of GT parts and remap them
    to labels 1..k (k <= KMAX). All other parts become background for that step.
    This forces the model to learn "seeded region growing" from CT + prompts.

Loss:
  - DiceCE (multi-class) over the dense target
  - Prompt consistency CE loss on ONLY the prompted voxels (so clicks strongly affect output)

Data layout (nnU-Net raw):
  <data_root>/
    imagesTr/
      case_0000_0000.nii.gz
    labelsTr/
      case_0000.nii.gz   (0=background, 1..N parts within this sample)

Run:
  python trainer_interactive_labeled_clicks_unet.py
"""

import os, glob, random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped,
    ScaleIntensityd, CropForegroundd, SpatialPadd, CenterSpatialCropd,
    RandFlipd, RandRotate90d,
)
from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from scipy import ndimage as ndi


# -------------------------
# 0) utils
# -------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_pairs(images_dir, labels_dir):
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    if len(imgs) == 0:
        raise RuntimeError(f"No NIfTI files found in {images_dir}")

    data = []
    for img in imgs:
        base = os.path.basename(img)
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


def _draw_ball_u8(mask_u8: np.ndarray, center_zyx, radius: int):
    """mask_u8: (D,H,W) uint8. center_zyx: (z,y,x)."""
    D, H, W = mask_u8.shape
    cz, cy, cx = center_zyx
    z0, z1 = max(0, cz - radius), min(D, cz + radius + 1)
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)

    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    ball = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    mask_u8[z0:z1, y0:y1, x0:x1][ball] = 1
    return mask_u8


# -------------------------
# 1) prompt simulation + remapped target
# -------------------------
def simulate_prompt_and_target_from_parts(
    parts_label_1dhw: torch.Tensor,
    kmax: int = 8,
    k_range=(2, 5),
    clicks_per_label=(1, 3),
    click_radius=(1, 3),
    p_empty=0.05,
):
    """
    parts_label_1dhw: (1,D,H,W) int. 0=bg, 1..N parts
    Returns:
      prompt_onehot: (kmax, D, H, W) float {0,1}
      target_int:   (D, H, W) long in [0..kmax] (dense training target)
      prompt_int:   (D, H, W) long in [0..kmax] (sparse prompt IDs at clicked voxels)
    Semantics:
      - We pick a random subset of GT parts and remap them to labels 1..k (k<=kmax).
      - Only those chosen parts are "foreground classes" for this iteration.
      - All other parts become background (0) for this iteration.
    """
    lbl = parts_label_1dhw[0].detach().cpu().numpy().astype(np.int32)  # (D,H,W)

    # optionally no prompts: train automatic mode too
    if np.random.rand() < p_empty:
        D, H, W = lbl.shape
        prompt_onehot = np.zeros((kmax, D, H, W), dtype=np.float32)
        prompt_int = np.zeros((D, H, W), dtype=np.int64)
        # In "no prompt" mode, we train object-vs-background union as class 1 (optional).
        # But that would introduce semantics. So instead we train "background only" which is useless.
        # Better: still do prompted training mostly. We'll simply return empty prompts and use
        # a remapped target with k=1 on union object so it learns an auto fallback without semantics:
        target_int = np.zeros((D, H, W), dtype=np.int64)
        target_int[lbl > 0] = 1
        # set channel for label 1 empty (no clicks), still ok
        return (
            torch.from_numpy(prompt_onehot),
            torch.from_numpy(target_int),
            torch.from_numpy(prompt_int),
        )

    part_ids = np.unique(lbl)
    part_ids = part_ids[part_ids != 0]
    if part_ids.size == 0:
        D, H, W = lbl.shape
        prompt_onehot = np.zeros((kmax, D, H, W), dtype=np.float32)
        prompt_int = np.zeros((D, H, W), dtype=np.int64)
        target_int = np.zeros((D, H, W), dtype=np.int64)
        return (
            torch.from_numpy(prompt_onehot),
            torch.from_numpy(target_int),
            torch.from_numpy(prompt_int),
        )

    # choose k parts
    k_hi = min(k_range[1], int(part_ids.size), kmax)
    k_lo = min(k_range[0], k_hi)
    k = np.random.randint(k_lo, k_hi + 1) if k_hi >= k_lo else k_hi
    chosen = np.random.choice(part_ids, size=k, replace=False)

    # map chosen GT part IDs -> prompt IDs 1..k
    mapping = {int(pid): (i + 1) for i, pid in enumerate(chosen)}

    D, H, W = lbl.shape
    target_int = np.zeros((D, H, W), dtype=np.int64)
    for pid, new_id in mapping.items():
        target_int[lbl == pid] = new_id

    # sparse prompt clicks (integer IDs)
    prompt_int = np.zeros((D, H, W), dtype=np.int64)
    for pid, new_id in mapping.items():
        vox = np.argwhere(lbl == pid)
        if vox.shape[0] == 0:
            continue
        n_clicks = np.random.randint(clicks_per_label[0], clicks_per_label[1] + 1)
        for _ in range(n_clicks):
            cz, cy, cx = vox[np.random.randint(vox.shape[0])]
            r = np.random.randint(click_radius[0], click_radius[1] + 1)
            tmp = np.zeros((D, H, W), dtype=np.uint8)
            _draw_ball_u8(tmp, (int(cz), int(cy), int(cx)), int(r))
            # write label id where the ball is
            prompt_int[tmp > 0] = new_id

    # one-hot encode prompt into kmax channels
    prompt_onehot = np.zeros((kmax, D, H, W), dtype=np.float32)
    for c in range(1, kmax + 1):
        prompt_onehot[c - 1] = (prompt_int == c).astype(np.float32)

    return (
        torch.from_numpy(prompt_onehot),      # (kmax,D,H,W)
        torch.from_numpy(target_int),         # (D,H,W)
        torch.from_numpy(prompt_int),         # (D,H,W)
    )


def prompt_click_consistency_ce(
    logits: torch.Tensor,
    prompt_int: torch.Tensor,
):
    """
    Enforce that on clicked voxels the predicted class matches the prompt ID.

    logits:     (B, C, D, H, W) where C = kmax+1
    prompt_int: (B, D, H, W) long in [0..kmax], sparse (>0 only at clicks)
    """
    # only where prompt_int > 0
    m = prompt_int > 0  # (B,D,H,W)
    if not m.any():
        return logits.new_tensor(0.0)

    # flatten selected voxels
    # logits: (B,C,D,H,W) -> (N,C)
    sel_logits = logits.permute(0, 2, 3, 4, 1)[m]   # (N,C)
    sel_target = prompt_int[m]                      # (N,)
    return F.cross_entropy(sel_logits, sel_target, reduction="mean")


# -------------------------
# 2) training loop
# -------------------------
def train(
    data_root: str,
    out_dir: str = "./runs_clickprompts",
    kmax: int = 8,
    patch_size=(64, 64, 64),
    batch_size: int = 1,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = None,
    cache_rate: float = 0.2,
    seed: int = 0,
    save_every: int = 25,
    val_every: int = 1,
    prompt_ce_w: float = 0.5,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    files = make_pairs(os.path.join(data_root, "imagesTr"), os.path.join(data_root, "labelsTr"))
    random.shuffle(files)

    if len(files) < 2:
        train_files = files
        val_files = files
    else:
        n_val = max(1, int(0.2 * len(files)))
        val_files = files[:n_val]
        train_files = files[n_val:] if len(files[n_val:]) > 0 else files

    print(f"Found total: {len(files)} | train: {len(train_files)} | val: {len(val_files)}")
    print(f"KMAX={kmax} -> in_channels={1+kmax}, out_channels={1+kmax}")

    tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),

        CropForegroundd(keys=["image", "label"], source_key="label"),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),

        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
    ])

    train_ds = CacheDataset(train_files, transform=tf, cache_rate=cache_rate, num_workers=0)
    val_ds   = CacheDataset(val_files,   transform=tf, cache_rate=cache_rate, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=list_data_collate)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)

    # UNet: multi-class
    model = UNet(
        spatial_dims=3,
        in_channels=1 + kmax,   # CT + kmax prompt one-hot channels
        out_channels=1 + kmax,  # background + kmax prompted labels
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Dice + CE combined, multi-class
    # - softmax=True because multi-class
    # - to_onehot_y=True expects y as integer (B,1,D,H,W) or (B,D,H,W) depending; we will give (B,1,...) below
    seg_loss = DiceCELoss(
        softmax=True,
        to_onehot_y=True,
        include_background=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = -1.0

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        epoch_loss = 0.0
        steps = 0

        for step, batch in enumerate(train_loader):
            ct = batch["image"].to(device)   # (B,1,D,H,W)
            gt_parts = batch["label"].to(device)  # (B,1,D,H,W) int parts

            # build prompt + remapped target per item
            prompt_oh_list = []
            target_list = []
            prompt_int_list = []

            for b in range(ct.shape[0]):
                prompt_oh, target_int, prompt_int = simulate_prompt_and_target_from_parts(
                    gt_parts[b],
                    kmax=kmax,
                    k_range=(2, 5),
                    clicks_per_label=(1, 3),
                    click_radius=(1, 3),
                    p_empty=0.05,
                )
                prompt_oh_list.append(prompt_oh)      # (kmax,D,H,W)
                target_list.append(target_int)        # (D,H,W)
                prompt_int_list.append(prompt_int)    # (D,H,W)

            prompt_oh = torch.stack(prompt_oh_list, dim=0).to(device)             # (B,kmax,D,H,W)
            target_int = torch.stack(target_list, dim=0).unsqueeze(1).to(device)  # (B,1,D,H,W)
            prompt_int = torch.stack(prompt_int_list, dim=0).to(device)           # (B,D,H,W)

            # concatenate input: CT + prompt_onehot
            x = torch.cat([ct, prompt_oh], dim=1)  # (B,1+kmax,D,H,W)

            opt.zero_grad(set_to_none=True)
            logits = model(x)  # (B,1+kmax,D,H,W)

            loss_seg = seg_loss(logits, target_int)
            loss_prompt = prompt_click_consistency_ce(logits, prompt_int)
            loss = loss_seg + prompt_ce_w * loss_prompt

            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().item())
            steps += 1

            if epoch == 1 and step == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)  # (B,D,H,W)
                    print("DEBUG shapes:",
                          "ct", tuple(ct.shape),
                          "prompt_oh", tuple(prompt_oh.shape),
                          "x", tuple(x.shape),
                          "logits", tuple(logits.shape),
                          "target_int", tuple(target_int.shape))
                    print("DEBUG prompted voxels ratio:",
                          float((prompt_int > 0).float().mean().item()))
                    print("DEBUG pred unique (first item):",
                          torch.unique(pred[0]).detach().cpu().numpy()[:20])

        epoch_loss /= max(1, steps)

        # ---------- VAL ----------
        mean_dice_fg = None
        if (epoch % val_every) == 0:
            model.eval()
            dices = []

            with torch.no_grad():
                for batch in val_loader:
                    ct = batch["image"].to(device)
                    gt_parts = batch["label"].to(device)

                    # simulate prompts on val too (interactive evaluation)
                    prompt_oh_list = []
                    target_list = []
                    prompt_int_list = []
                    for b in range(ct.shape[0]):
                        prompt_oh, target_int, prompt_int = simulate_prompt_and_target_from_parts(
                            gt_parts[b],
                            kmax=kmax,
                            k_range=(2, 5),
                            clicks_per_label=(1, 3),
                            click_radius=(1, 3),
                            p_empty=0.05,
                        )
                        prompt_oh_list.append(prompt_oh)
                        target_list.append(target_int)
                        prompt_int_list.append(prompt_int)

                    prompt_oh = torch.stack(prompt_oh_list, dim=0).to(device)
                    target_int = torch.stack(target_list, dim=0).to(device)  # (B,D,H,W)
                    x = torch.cat([ct, prompt_oh], dim=1)

                    logits = model(x)
                    pred = torch.argmax(logits, dim=1)  # (B,D,H,W)

                    # Evaluate foreground union dice: (pred>0) vs (target>0)
                    pred_fg = (pred > 0).float()
                    tgt_fg = (target_int > 0).float()
                    inter = (pred_fg * tgt_fg).sum(dim=(1, 2, 3))
                    denom = pred_fg.sum(dim=(1, 2, 3)) + tgt_fg.sum(dim=(1, 2, 3))
                    dice = (2 * inter / (denom + 1e-8))
                    dices.append(float(dice.mean().item()))

            mean_dice_fg = float(np.mean(dices)) if len(dices) else 0.0

        msg = f"Epoch {epoch:03d} | loss={epoch_loss:.4f}"
        if mean_dice_fg is not None:
            msg += f" | val_fg_union_dice={mean_dice_fg:.4f}"
        print(msg)

        # ---------- SAVE ----------
        if (epoch % save_every) == 0 or epoch == epochs:
            ckpt_path = os.path.join(out_dir, f"model_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("Saved:", ckpt_path)

        if mean_dice_fg is not None and mean_dice_fg > best_val:
            best_val = mean_dice_fg
            best_path = os.path.join(out_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best val={best_val:.4f} -> Saved:", best_path)


if __name__ == "__main__":
    train(
        data_root=r"C:/uniDev/forschungsprojekt/trainingsdata/nnUNet_raw/Dataset001_CADSynthetic",
        out_dir=r"./runs_clickprompts",
        kmax=8,
        patch_size=(64, 64, 64),
        batch_size=1,
        epochs=300,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_rate=0.2,
        seed=0,
        save_every=25,
        val_every=1,
        prompt_ce_w=0.5,  # tune 0.2..1.0
    )
