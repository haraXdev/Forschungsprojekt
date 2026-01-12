"""
trainer_instance_click_unet_full.py

This version FIXES your current script so it works with your data where:
- each CT has variable instance ids (0..N), and ids are independent across CTs
- therefore "num_classes" / "K click channels" is NOT valid

What it trains:
    (CT + click prompt) -> binary mask of the clicked instance

Input channels: 2  (CT + pos-click map)
Output channels: 2 (background vs object)

Data layout (nnU-Net raw, as you generated):
  <data_root>/
    imagesTr/
      case_0000_0000.nii.gz
      case_0001_0000.nii.gz
      ...
    labelsTr/
      case_0000.nii.gz
      case_0001.nii.gz
      ...

Run:
  python trainer_instance_click_unet_full.py
"""

import os, glob, random
import torch
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    EnsureTyped, RandFlipd, RandRotate90d, RandCropByPosNegLabeld,
    ScaleIntensityd
)
from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


# -------------------------
# 1) Click-map generation
# -------------------------
def gaussian_map_3d(shape, points, sigma, device):
    """shape=(D,H,W), points=[(z,y,x), ...] -> (D,H,W) torch float"""
    D, H, W = shape
    if len(points) == 0:
        return torch.zeros((D, H, W), device=device, dtype=torch.float32)

    zz = torch.arange(D, device=device).view(D, 1, 1)
    yy = torch.arange(H, device=device).view(1, H, 1)
    xx = torch.arange(W, device=device).view(1, 1, W)

    out = torch.zeros((D, H, W), device=device, dtype=torch.float32)
    denom = 2.0 * (sigma ** 2)

    for (z, y, x) in points:
        g = torch.exp(-((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) / denom)
        out = torch.maximum(out, g)  # multiple clicks: keep max
    return out


# -------------------------
# 2) Instance click sampler
# -------------------------
class InstanceClickSampler(torch.nn.Module):
    """
    Creates training pairs for instance-based interactive segmentation.

    Input:
      image: (1, D, H, W)  float
      label: (1, D, H, W)  int instance ids (0..N), N varies per volume

    Output:
      x: (2, D, H, W)   = [CT, click_map]
      y: (1, D, H, W)   = binary target for ONE chosen instance (0/1)
    """

    def __init__(self, sigma: float = 3.0, max_clicks: int = 3, p_no_click: float = 0.0):
        super().__init__()
        self.sigma = float(sigma)
        self.max_clicks = int(max_clicks)
        self.p_no_click = float(p_no_click)

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        device = image.device
        lbl = label[0].long()  # (D,H,W)

        # Find instance ids present (ignore background=0)
        ids = torch.unique(lbl)
        ids = ids[ids != 0]
        if ids.numel() == 0:
            return None, None  # no object in this patch

        # Pick one instance id from this patch
        t = ids[torch.randint(0, ids.numel(), (1,), device=device)].item()

        # Binary GT for the chosen instance
        y = (lbl == t).long().unsqueeze(0)  # (1,D,H,W) with {0,1}

        # Optionally simulate "no clicks"
        if self.p_no_click > 0.0 and random.random() < self.p_no_click:
            D, H, W = y.shape[1:]
            click = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
            x = torch.cat([image, click], dim=0)  # (2,D,H,W)
            return x, y

        # Sample 1..max_clicks clicks inside chosen instance
        idx = (y[0] == 1).nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None, None

        n = random.randint(1, self.max_clicks)
        sel = idx[torch.randint(0, idx.shape[0], (n,), device=device)]
        pts = [(int(z), int(y_), int(x)) for z, y_, x in sel]

        D, H, W = y.shape[1:]
        click_map = gaussian_map_3d((D, H, W), pts, sigma=self.sigma, device=device).unsqueeze(0)

        # Final net input: CT + click map
        x = torch.cat([image, click_map], dim=0)  # (2,D,H,W)
        return x, y


# -------------------------
# 3) Dataset listing (nnU-Net naming)
# -------------------------
def make_pairs(images_dir, labels_dir):
    """
    imagesTr and labelsTr contain IDENTICAL filenames:
      imagesTr/0000_0000.nii.gz
      labelsTr/0000_0000.nii.gz
    """
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
    if len(imgs) == 0:
        raise RuntimeError(f"No NIfTI files found in {images_dir}")

    data = []
    for img in imgs:
        base = os.path.basename(img)
        lbl = os.path.join(labels_dir, base)

        if not os.path.exists(lbl):
            raise FileNotFoundError(f"Label missing for {img}: expected {lbl}")

        data.append({"image": img, "label": lbl})

    return data


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
    sigma=3.0,
    max_clicks=3,
    p_no_click=0.0,
    cache_rate=0.2,
):
    train_files = make_pairs(
        os.path.join(data_root, "imagesTr"),
        os.path.join(data_root, "labelsTr"),
    )
    if len(train_files) == 0:
        raise RuntimeError("No training files found. Check your data_root/imagesTr path.")

    # Split (never make train empty)
    random.shuffle(train_files)
    if len(train_files) < 2:
        val_files = train_files
        train_split = train_files
    else:
        n_val = max(1, int(0.2 * len(train_files)))
        val_files = train_files[:n_val]
        train_split = train_files[n_val:] if len(train_files[n_val:]) > 0 else train_files

    print(f"Found total: {len(train_files)} | train: {len(train_split)} | val: {len(val_files)}")

    base_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3, neg=1,
            num_samples=2,     # <--- more patches per volume helps a lot
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    ])

    train_ds = CacheDataset(train_split, transform=base_tf, cache_rate=cache_rate, num_workers=0)
    val_ds   = CacheDataset(val_files,  transform=base_tf, cache_rate=cache_rate, num_workers=0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=list_data_collate,
    )

    # Binary model: BG vs clicked instance
    model = UNet(
        spatial_dims=3,
        in_channels=2,   # CT + click map
        out_channels=2,  # BG vs OBJ
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    click_sampler = InstanceClickSampler(sigma=sigma, max_clicks=max_clicks, p_no_click=p_no_click)

    best = -1.0

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            img = batch["image"].to(device)  # (B,1,D,H,W)
            lbl = batch["label"].to(device)  # (B,1,D,H,W)

            xs, ys = [], []
            for b in range(img.shape[0]):
                x, y = click_sampler(img[b], lbl[b])
                if x is None:
                    continue
                xs.append(x)
                ys.append(y)

            if len(xs) == 0:
                continue

            x = torch.stack(xs, dim=0)  # (B,2,D,H,W)
            y = torch.stack(ys, dim=0)  # (B,1,D,H,W)

            opt.zero_grad(set_to_none=True)
            logits = model(x)           # (B,2,D,H,W)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            epoch_loss += loss.detach().item()
            steps += 1

        epoch_loss /= max(1, steps)

        # ---------------- VAL ----------------
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)

                xs, ys = [], []
                for b in range(img.shape[0]):
                    x, y = click_sampler(img[b], lbl[b])
                    if x is None:
                        continue
                    xs.append(x)
                    ys.append(y)

                if len(xs) == 0:
                    continue

                x = torch.stack(xs, dim=0)
                y = torch.stack(ys, dim=0)

                pred = torch.softmax(model(x), dim=1)  # (B,2,D,H,W)
                dice_metric(y_pred=pred, y=y)

            mean_dice = float(dice_metric.aggregate().item())

        print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | val_dice={mean_dice:.4f}")

        if mean_dice > best:
            best = mean_dice
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_dice": best},
                "best_instance_click_unet.pth"
            )

    print(f"Best val dice: {best:.4f} (saved to best_instance_click_unet.pth)")


if __name__ == "__main__":
    train(
        data_root=r"C:/uniDev/forschungsprojekt/trainingsdata/nnUNet_raw/Dataset001_CADSynthetic",
        patch_size=(128, 128, 128),
        batch_size=1,
        epochs=50,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sigma=3.0,
        max_clicks=3,
        p_no_click=0.0,
        cache_rate=0.2,
    )
