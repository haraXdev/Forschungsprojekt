import os, glob, random
import numpy as np
import torch
from monai.transforms import ScaleIntensityd
from torch.utils.data import DataLoader
from monai.data import list_data_collate

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, RandFlipd, RandRotate90d,
    RandCropByPosNegLabeld
)
from monai.data import CacheDataset, Dataset
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


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


class AddClickChannels(torch.nn.Module):
    """
    Adds K click channels based on GT labels:
      input: image (1,D,H,W), label (1,D,H,W) with ints {0..K}
      output: image_with_clicks (1+K,D,H,W), label unchanged
    """
    def __init__(self, num_classes: int, sigma: float = 3.0, max_clicks_per_class: int = 3,
                 p_zero_click_class: float = 0.2):
        super().__init__()
        self.K = num_classes
        self.sigma = sigma
        self.max_clicks = max_clicks_per_class
        self.p_zero = p_zero_click_class

    def forward(self, image: torch.Tensor, label: torch.Tensor):
        # image: (1,D,H,W), label: (1,D,H,W)
        device = image.device
        _, D, H, W = image.shape

        lbl = label[0].long()  # (D,H,W)
        click_ch = []

        for k in range(1, self.K + 1):
            # sometimes simulate "no clicks for this class"
            if random.random() < self.p_zero:
                pts = []
            else:
                idx = (lbl == k).nonzero(as_tuple=False)
                if idx.numel() == 0:
                    pts = []
                else:
                    n = random.randint(1, self.max_clicks)
                    # sample points from voxels of class k
                    sel = idx[torch.randint(0, idx.shape[0], (n,))]
                    pts = [(int(z), int(y), int(x)) for z, y, x in sel]

            m = gaussian_map_3d((D, H, W), pts, sigma=self.sigma, device=device)  # (D,H,W)
            click_ch.append(m)

        clicks = torch.stack(click_ch, dim=0)  # (K,D,H,W)
        out = torch.cat([image, clicks], dim=0)  # (1+K,D,H,W)
        return out, label


# -------------------------
# 2) Dataset listing
# -------------------------
def make_pairs(images_dir, labels_dir):
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz*")))
    data = []
    for img in imgs:
        base = os.path.basename(img)
        lbl = os.path.join(labels_dir, base)
        if not os.path.exists(lbl):
            raise FileNotFoundError(f"Label missing for {img}: expected {lbl}")
        data.append({"image": img, "label": lbl})
    return data


# -------------------------
# 3) Main training
# -------------------------
def train(
    data_root="data",
    num_classes=4,              # K (labels 1..K)
    patch_size=(128, 128, 128),
    batch_size=1,
    epochs=100,
    lr=1e-4,
    device="cuda"
):
    train_files = make_pairs(f"{data_root}/imagesTr", f"{data_root}/labelsTr")

    # simple split
    random.shuffle(train_files)
    n_val = max(1, int(0.2 * len(train_files)))
    val_files = train_files[:n_val]
    train_files = train_files[n_val:]

    # base transforms (load + normalize + patch crop)
    base_tf = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # optional: set spacing if your industrial CT needs it; otherwise remove Spacingd
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),          
        EnsureTyped(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1, neg=1,
            num_samples=1,   # one patch per volume per iteration
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    ])

    # datasets
    train_ds = CacheDataset(train_files, transform=base_tf, cache_rate=0.2, num_workers=2)
    val_ds   = CacheDataset(val_files,   transform=base_tf, cache_rate=0.2, num_workers=2)

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
    # model
    model = UNet(
        spatial_dims=3,
        in_channels=1 + num_classes,   # CT + K click channels
        out_channels=num_classes + 1,  # BG + K
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    click_adder = AddClickChannels(
        num_classes=num_classes,
        sigma=3.0,
        max_clicks_per_class=3,
        p_zero_click_class=0.3,   # wichtig: lernt auch mit "fehlenden" clicks klarzukommen
    )

    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            img = batch["image"].to(device)  # (B,1,D,H,W)
            lbl = batch["label"].to(device)  # (B,1,D,H,W)

            # add clicks per sample in batch
            xs, ys = [], []
            for b in range(img.shape[0]):
                x, y = click_adder(img[b], lbl[b])
                xs.append(x)
                ys.append(y)
            x = torch.stack(xs, dim=0)  # (B,1+K,D,H,W)
            y = torch.stack(ys, dim=0)  # (B,1,D,H,W)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            epoch_loss += float(loss)

        epoch_loss /= max(1, len(train_loader))

        # validation (simple)
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)

                # For val: you can simulate a few clicks too (or set all to zero).
                xs = []
                for b in range(img.shape[0]):
                    x, _ = click_adder(img[b], lbl[b])
                    xs.append(x)
                x = torch.stack(xs, dim=0)

                # patch-based inference if needed (optional for small patches)
                pred = model(x)
                pred = torch.softmax(pred, dim=1)

                dice_metric(y_pred=pred, y=lbl)

            mean_dice = dice_metric.aggregate().item()

        print(f"Epoch {epoch:03d} | loss={epoch_loss:.4f} | val_dice={mean_dice:.4f}")

        if mean_dice > best:
            best = mean_dice
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_dice": best},
                "best_click_unet.pth"
            )

    print(f"Best val dice: {best:.4f} (saved to best_click_unet.pth)")


if __name__ == "__main__":
    train(
        data_root="C:/uniDev/forschungsprojekt/trainingsdata/nnUNet_raw/Dataset001_CADSynthetic",  # <-- set your data path here
        num_classes=10,        # <-- set K here
        patch_size=(128,128,128),
        batch_size=1,
        epochs=50,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
