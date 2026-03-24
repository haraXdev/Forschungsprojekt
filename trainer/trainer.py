"""
trainer_interactive_labeled_prompts_unet_zarr.py
"""

import os
os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"
import random
import numpy as np
import zarr
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
)
from monai.data import Dataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceCELoss


# -------------------------
# 0) utils
# -------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_pairs(data_root):
    case_folders = [
        os.path.join(data_root, d)
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
    ]

    if len(case_folders) == 0:
        raise RuntimeError(f"No case folders found in {data_root}")

    data = []
    for case_folder in case_folders:
        case_path = os.path.join(case_folder, "data.zarr")

        if not os.path.isdir(case_path):
            raise RuntimeError(f"Missing data.zarr in {case_folder}")

        print(f"Loading case: {case_path}")

        
        store = zarr.open_group(case_path, mode="r", zarr_version=3)

        ct_vol  = store["input"][:].astype(np.float32)
        lbl_vol = store["label"][:].astype(np.int64)

        print("  image shape:", ct_vol.shape, "dtype:", ct_vol.dtype)
        print("  label shape:", lbl_vol.shape, "dtype:", lbl_vol.dtype)

        data.append({
            "image": ct_vol,
            "label": lbl_vol,
            "case_path": case_path,
        })

    return data


def draw_napari_click_u8(
    mask_u8: np.ndarray,
    center_zyx,
    brush_size: int = 4,
    z_thickness: int = 0,
    jitter: float = 0.2,
    fray: float = 0.08,
):
    D, H, W = mask_u8.shape
    cz, cy, cx = center_zyx

    r = brush_size / 2.0

    cy = float(cy) + np.random.uniform(-jitter, jitter)
    cx = float(cx) + np.random.uniform(-jitter, jitter)
    cz = int(round(float(cz)))

    z0 = max(0, cz - z_thickness)
    z1 = min(D, cz + z_thickness + 1)

    for z in range(z0, z1):
        y0 = max(0, int(np.floor(cy - r - 1)))
        y1 = min(H, int(np.ceil(cy + r + 2)))
        x0 = max(0, int(np.floor(cx - r - 1)))
        x1 = min(W, int(np.ceil(cx + r + 2)))

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        disk = dist2 <= (r ** 2)

        if fray > 0:
            ring = (dist2 >= max(0.0, (r - 0.75) ** 2)) & (dist2 <= (r + 0.75) ** 2)
            drop = (np.random.rand(*disk.shape) < fray) & ring
            disk = disk & (~drop)

        mask_u8[z, y0:y1, x0:x1][disk] = 1

    return mask_u8


def _disk_offsets(radius: float):
    r = int(np.ceil(radius))
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= radius * radius:
                offs.append((dy, dx))
    return offs


def _paint_disk_on_slice(mask_u8: np.ndarray, z: int, y: float, x: float, radius: float):
    D, H, W = mask_u8.shape
    z = int(z)
    if not (0 <= z < D):
        return

    cy = int(round(y))
    cx = int(round(x))
    for dy, dx in _disk_offsets(radius):
        yy = cy + dy
        xx = cx + dx
        if 0 <= yy < H and 0 <= xx < W:
            mask_u8[z, yy, xx] = 1


def _sample_point_from_mask_2d(mask_2d: np.ndarray):
    ys, xs = np.where(mask_2d)
    if len(ys) == 0:
        return None
    i = np.random.randint(len(ys))
    return float(ys[i]), float(xs[i])


def _sample_large_slice_for_part(
    part_mask: np.ndarray,
    min_area_ratio: float = 0.6,
    temperature: float = 1.2,
):
    areas = part_mask.sum(axis=(1, 2)).astype(np.float32)
    max_area = areas.max()
    if max_area <= 0:
        return None

    eligible = np.where(areas >= (min_area_ratio * max_area))[0]
    if eligible.size == 0:
        return int(np.argmax(areas))

    weights = areas[eligible].copy()
    weights = np.maximum(weights, 1e-8)

    if temperature != 1.0:
        weights = np.power(weights, 1.0 / max(temperature, 1e-8))

    weights = weights / weights.sum()
    z = np.random.choice(eligible, p=weights)
    return int(z)


def draw_scribble_u8_on_part(
    mask_u8: np.ndarray,
    part_mask: np.ndarray,
    brush_size: int = 4,
    n_segments_range=(8, 14),
    step_range=(2.0, 4.0),
    turn_std: float = 0.6,
    max_retry_per_step: int = 12,
    min_area_ratio: float = 0.6,
    slice_temperature: float = 1.2,
):
    D, H, W = mask_u8.shape
    _ = D, H, W
    part_mask = part_mask.astype(bool)

    z = _sample_large_slice_for_part(
        part_mask,
        min_area_ratio=min_area_ratio,
        temperature=slice_temperature,
    )
    if z is None:
        return mask_u8

    surf = part_mask[z]
    start = _sample_point_from_mask_2d(surf)
    if start is None:
        return mask_u8

    y, x = start
    radius = brush_size / 2.0

    theta = np.random.uniform(0, 2 * np.pi)
    n_segments = np.random.randint(n_segments_range[0], n_segments_range[1] + 1)

    _paint_disk_on_slice(mask_u8, z, y, x, radius)

    for _ in range(n_segments):
        success = False

        for _retry in range(max_retry_per_step):
            theta_try = theta + np.random.normal(0.0, turn_std)
            step = np.random.uniform(step_range[0], step_range[1])

            ny = y + step * np.sin(theta_try)
            nx = x + step * np.cos(theta_try)

            iy = int(round(ny))
            ix = int(round(nx))

            if 0 <= iy < surf.shape[0] and 0 <= ix < surf.shape[1] and surf[iy, ix]:
                dist = max(1, int(np.ceil(np.hypot(ny - y, nx - x))))
                for t in np.linspace(0.0, 1.0, dist + 1):
                    py = y * (1 - t) + ny * t
                    px = x * (1 - t) + nx * t
                    ipy = int(round(py))
                    ipx = int(round(px))
                    if 0 <= ipy < surf.shape[0] and 0 <= ipx < surf.shape[1] and surf[ipy, ipx]:
                        _paint_disk_on_slice(mask_u8, z, py, px, radius)

                y, x = ny, nx
                theta = theta_try
                success = True
                break

        if not success:
            break

    return mask_u8


# -------------------------
# 1) prompt simulation + remapped target
# -------------------------
def simulate_prompt_and_target_from_parts(
    parts_label_1dhw: torch.Tensor,
    kmax: int = 8,
    k_range=(2, 5),
    clicks_per_label=(3, 6),
    brush_size: int = 4,
    p_empty: float = 0.05,
    scribble_prob: float = 0.6,
    scribble_segments_range=(8, 14),
    scribble_step_range=(2.0, 4.0),
    scribble_min_area_ratio: float = 0.6,
    scribble_slice_temperature: float = 1.2,
):
    lbl = parts_label_1dhw[0].detach().cpu().numpy().astype(np.int32)

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

    k_hi = min(k_range[1], int(part_ids.size), kmax)
    k_lo = min(k_range[0], k_hi)
    k = np.random.randint(k_lo, k_hi + 1) if k_hi >= k_lo else k_hi
    chosen = np.random.choice(part_ids, size=k, replace=False)

    mapping = {int(pid): (i + 1) for i, pid in enumerate(chosen)}

    D, H, W = lbl.shape
    target_int = np.zeros((D, H, W), dtype=np.int64)
    for pid, new_id in mapping.items():
        target_int[lbl == pid] = new_id

    prompt_int = np.zeros((D, H, W), dtype=np.int64)

    for pid, new_id in mapping.items():
        part_mask = (lbl == pid)
        vox = np.argwhere(part_mask)
        if vox.shape[0] == 0:
            continue

        n_clicks = np.random.randint(clicks_per_label[0], clicks_per_label[1] + 1)
        for _ in range(n_clicks):
            cz, cy, cx = vox[np.random.randint(vox.shape[0])]
            tmp = np.zeros((D, H, W), dtype=np.uint8)
            draw_napari_click_u8(
                tmp,
                (int(cz), int(cy), int(cx)),
                brush_size=brush_size,
            )
            tmp = tmp & part_mask.astype(np.uint8)
            prompt_int[tmp > 0] = new_id

        if np.random.rand() < scribble_prob:
            tmp = np.zeros((D, H, W), dtype=np.uint8)
            draw_scribble_u8_on_part(
                tmp,
                part_mask=part_mask,
                brush_size=brush_size,
                n_segments_range=scribble_segments_range,
                step_range=scribble_step_range,
                turn_std=0.6,
                min_area_ratio=scribble_min_area_ratio,
                slice_temperature=scribble_slice_temperature,
            )
            tmp = tmp & part_mask.astype(np.uint8)
            prompt_int[tmp > 0] = new_id

    prompt_onehot = np.zeros((kmax, D, H, W), dtype=np.float32)
    for c in range(1, kmax + 1):
        prompt_onehot[c - 1] = (prompt_int == c).astype(np.float32)

    return (
        torch.from_numpy(prompt_onehot),
        torch.from_numpy(target_int),
        torch.from_numpy(prompt_int),
    )


def prompt_click_consistency_ce(
    logits: torch.Tensor,
    prompt_int: torch.Tensor,
):
    m = prompt_int > 0
    if not m.any():
        return logits.new_tensor(0.0)

    sel_logits = logits.permute(0, 2, 3, 4, 1)[m]
    sel_target = prompt_int[m]
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
    val_every: int = 10,
    prompt_ce_w: float = 0.5,
    clicks_per_label=(3, 6),
    brush_size: int = 4,
    p_empty: float = 0.05,
    scribble_prob: float = 0.6,
    scribble_segments_range=(8, 14),
    scribble_step_range=(2.0, 4.0),
    scribble_min_area_ratio: float = 0.6,
    scribble_slice_temperature: float = 1.2,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    lr_min: float = 1e-6,
    lr_threshold: float = 1e-4,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    tb_dir = os.path.join(out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text(
        "hparams",
        f"kmax={kmax}, patch_size={patch_size}, init_lr={lr}, prompt_ce_w={prompt_ce_w}, "
        f"clicks_per_label={clicks_per_label}, brush_size={brush_size}, p_empty={p_empty}, "
        f"scribble_prob={scribble_prob}, scribble_segments_range={scribble_segments_range}, "
        f"scribble_step_range={scribble_step_range}, scribble_min_area_ratio={scribble_min_area_ratio}, "
        f"scribble_slice_temperature={scribble_slice_temperature}, "
        f"ReduceLROnPlateau(monitor=val_loss, factor={lr_factor}, patience={lr_patience}, min_lr={lr_min})"
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    files = make_pairs(data_root)
    random.shuffle(files)

    if len(files) < 2:
        train_files = files
        val_files = files
    else:
        n_val = max(1, int(0.2 * len(files)))
        val_files = files[:n_val]
        train_files = files[n_val:] if len(files[n_val:]) > 0 else files

    print(f"Found total: {len(files)} | train: {len(train_files)} | val: {len(val_files)}")
    print(f"KMAX={kmax} -> in_channels={1 + kmax}, out_channels={1 + kmax}")

    tf = Compose([
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image"], dtype=torch.float32),
    EnsureTyped(keys=["label"], dtype=torch.int64),

    CropForegroundd(keys=["image", "label"], source_key="label"),
    SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
])

    train_ds = Dataset(train_files, transform=tf)
    val_ds = Dataset(val_files, transform=tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1 + kmax,
        out_channels=1 + kmax,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    seg_loss = DiceCELoss(
        softmax=True,
        to_onehot_y=True,
        include_background=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_threshold,
        threshold_mode="rel",
        cooldown=0,
        min_lr=lr_min,
    )

    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for step, batch in enumerate(train_loader):
            ct = batch["image"].to(device)
            gt_parts = batch["label"].to(device)

            prompt_oh_list = []
            target_list = []
            prompt_int_list = []

            for b in range(ct.shape[0]):
                prompt_oh, target_int, prompt_int = simulate_prompt_and_target_from_parts(
                    gt_parts[b],
                    kmax=kmax,
                    k_range=(2, 5),
                    clicks_per_label=clicks_per_label,
                    brush_size=brush_size,
                    p_empty=p_empty,
                    scribble_prob=scribble_prob,
                    scribble_segments_range=scribble_segments_range,
                    scribble_step_range=scribble_step_range,
                    scribble_min_area_ratio=scribble_min_area_ratio,
                    scribble_slice_temperature=scribble_slice_temperature,
                )
                prompt_oh_list.append(prompt_oh)
                target_list.append(target_int)
                prompt_int_list.append(prompt_int)

            prompt_oh = torch.stack(prompt_oh_list, dim=0).to(device)
            target_int = torch.stack(target_list, dim=0).unsqueeze(1).to(device)
            prompt_int = torch.stack(prompt_int_list, dim=0).to(device)

            x = torch.cat([ct, prompt_oh], dim=1)

            opt.zero_grad(set_to_none=True)
            logits = model(x)

            loss_seg = seg_loss(logits, target_int)
            loss_prompt = prompt_click_consistency_ce(logits, prompt_int)
            loss = loss_seg + prompt_ce_w * loss_prompt

            loss.backward()
            opt.step()

            if step % 20 == 0:
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

            train_loss_sum += loss.item()

            if epoch == 1 and step == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    print(
                        "DEBUG shapes:",
                        "ct", tuple(ct.shape),
                        "prompt_oh", tuple(prompt_oh.shape),
                        "x", tuple(x.shape),
                        "logits", tuple(logits.shape),
                        "target_int", tuple(target_int.shape),
                    )
                    print("DEBUG prompted voxels ratio:", float((prompt_int > 0).float().mean().item()))
                    print("DEBUG pred unique (first item):", torch.unique(pred[0]).detach().cpu().numpy()[:20])

        mean_train_loss = train_loss_sum / max(1, len(train_loader))
        writer.add_scalar("loss/train", mean_train_loss, epoch)

        current_lr = opt.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        mean_dice_fg = None
        mean_val_loss = None

        if (epoch % val_every) == 0:
            model.eval()
            val_loss_sum = 0.0
            dice_sum = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    ct = batch["image"].to(device)
                    gt_parts = batch["label"].to(device)

                    prompt_oh_list = []
                    target_list = []
                    prompt_int_list = []

                    for b in range(ct.shape[0]):
                        prompt_oh, target_int, prompt_int = simulate_prompt_and_target_from_parts(
                            gt_parts[b],
                            kmax=kmax,
                            k_range=(2, 5),
                            clicks_per_label=clicks_per_label,
                            brush_size=brush_size,
                            p_empty=p_empty,
                            scribble_prob=scribble_prob,
                            scribble_segments_range=scribble_segments_range,
                            scribble_step_range=scribble_step_range,
                            scribble_min_area_ratio=scribble_min_area_ratio,
                            scribble_slice_temperature=scribble_slice_temperature,
                        )
                        prompt_oh_list.append(prompt_oh)
                        target_list.append(target_int)
                        prompt_int_list.append(prompt_int)

                    prompt_oh = torch.stack(prompt_oh_list, dim=0).to(device)
                    target_int = torch.stack(target_list, dim=0).to(device)
                    x = torch.cat([ct, prompt_oh], dim=1)

                    logits = model(x)

                    val_target_int = target_int.unsqueeze(1)
                    loss_v = seg_loss(logits, val_target_int)
                    val_loss_sum += loss_v.item()

                    pred = torch.argmax(logits, dim=1)

                    pred_fg = (pred > 0).float()
                    tgt_fg = (target_int > 0).float()
                    inter = (pred_fg * tgt_fg).sum(dim=(1, 2, 3))
                    denom = pred_fg.sum(dim=(1, 2, 3)) + tgt_fg.sum(dim=(1, 2, 3))
                    dice = (2 * inter / (denom + 1e-8))

                    dice_sum += dice.mean().item()
                    n_val_batches += 1

            mean_val_loss = val_loss_sum / max(1, n_val_batches)
            mean_dice_fg = dice_sum / max(1, n_val_batches)

            writer.add_scalar("loss/val", mean_val_loss, epoch)
            writer.add_scalar("dice/val_fg_union", mean_dice_fg, epoch)

            scheduler.step(mean_val_loss)

            current_lr = opt.param_groups[0]["lr"]
            writer.add_scalar("lr_after_val", current_lr, epoch)

        msg = f"Epoch {epoch:03d} | train_loss={mean_train_loss:.4f} | lr={opt.param_groups[0]['lr']:.2e}"
        if mean_dice_fg is not None:
            msg += f" | val_loss={mean_val_loss:.4f} | val_fg_union_dice={mean_dice_fg:.4f}"
        print(msg)

        if (epoch % save_every) == 0 or epoch == epochs:
            ckpt_path = os.path.join(out_dir, f"model_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("Saved:", ckpt_path)

        if mean_dice_fg is not None and mean_dice_fg > best_val:
            best_val = mean_dice_fg
            best_path = os.path.join(out_dir, "model_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best val={best_val:.4f} -> Saved:", best_path)

    writer.close()


if __name__ == "__main__":
    train(
        data_root=r"/lgrp/edu-2025-2-brprj-segmentation/Forschungsprojekt/trainingsdata/Datasets/Dataset002_CT_Scans",
        out_dir=r"./runs_clickprompts",
        kmax=8,
        patch_size=(64, 64, 64),
        batch_size=1,
        epochs=500,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_rate=1,
        seed=0,
        save_every=25,
        val_every=10,
        prompt_ce_w=0.5,
        clicks_per_label=(3, 6),
        brush_size=4,
        p_empty=0.05,
        scribble_prob=0.6,
        scribble_segments_range=(8, 14),
        scribble_step_range=(2.0, 4.0),
        scribble_min_area_ratio=0.6,
        scribble_slice_temperature=1.2,
        lr_factor=0.5,
        lr_patience=3,
        lr_min=1e-6,
        lr_threshold=1e-4,
    )