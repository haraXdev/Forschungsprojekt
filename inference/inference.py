import os
from pathlib import Path

# ============================================================
# HARD STARTUP DEBUG (to prove which file is running + where)
# ============================================================
print("=== START inference.py (UPDATED DEBUG) ===", flush=True)
print("SCRIPT PATH:", Path(__file__).resolve(), flush=True)
print("CWD:", Path.cwd().resolve(), flush=True)

# Force debug log to live NEXT TO THIS SCRIPT (absolute path)
DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug_log.txt"
DEBUG_LOG_PATH.write_text("=== debug log created at startup ===\n", encoding="utf-8")
print("DEBUG_LOG_PATH:", DEBUG_LOG_PATH, flush=True)
# ============================================================

import numpy as np
import torch
import napari
from magicgui import magicgui
import nibabel as nib

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from napari.utils.notifications import show_info
from scipy import ndimage as ndi  # <-- FIX 1 needs this


# ============================================================
# CONFIG
# ============================================================
MODEL_CHECKPOINT_PATH = Path(r"C:/uniDev/fProject/inference/model/model_epoch_075.pth")
INPUT_IMAGE_PATH      = Path(r"C:/uniDev/fProject/inference/image/case_0005_0000.nii")

KMAX = 8
PATCH_SIZE = (64, 64, 64)      # must match training patch_size
SW_BATCH_SIZE = 2              # adjust for your VRAM
OVERLAP = 0.25                 # typical
FORCE_CPU = False

# Prompt-ROI settings (recommended for your training distribution)
PROMPT_MARGIN = 24             # voxels around prompts
DEFAULT_DILATE_RADIUS = 5      # <-- start with 5, try 4..8
# ============================================================


def log(msg: str):
    print(msg, flush=True)
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build_model(kmax: int = 8):
    return UNet(
        spatial_dims=3,
        in_channels=1 + kmax,
        out_channels=1 + kmax,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    )


def load_state_dict_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    state = torch.load(str(ckpt_path), map_location=device)

    # allow nested formats
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    # strip common outer prefixes if present
    cleaned = {}
    for k, v in state.items():
        for prefix in ["model.", "net.", "network."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    # ---- AUTO-FIX PREFIX MISMATCH ----
    model_keys = list(model.state_dict().keys())
    if not model_keys:
        raise RuntimeError("Model has no state_dict keys?")

    expects_model_prefix = model_keys[0].startswith("model.")
    ckpt_has_model_prefix = next(iter(cleaned.keys())).startswith("model.")

    if expects_model_prefix and not ckpt_has_model_prefix:
        cleaned = {f"model.{k}": v for k, v in cleaned.items()}

    if (not expects_model_prefix) and ckpt_has_model_prefix:
        cleaned = {k[len("model."):]: v for k, v in cleaned.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log(f"[WARN] Missing keys (up to 10): {missing[:10]}")
    if unexpected:
        log(f"[WARN] Unexpected keys (up to 10): {unexpected[:10]}")

    model.to(device).eval()
    return model


def load_nifti_as_ras_zyx(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    vol_xyz = img.get_fdata().astype(np.float32)   # (X,Y,Z)
    vol_zyx = np.transpose(vol_xyz, (2, 1, 0))     # (Z,Y,X)
    return vol_zyx


def scale_intensity_minmax_like_monai(vol: np.ndarray) -> np.ndarray:
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax <= vmin + 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol - vmin) / (vmax - vmin)
    return out.astype(np.float32)


def prompt_int_to_onehot(prompt_int_zyx: np.ndarray, kmax: int) -> np.ndarray:
    oh = np.zeros((kmax,) + prompt_int_zyx.shape, dtype=np.float32)
    for c in range(1, kmax + 1):
        oh[c - 1] = (prompt_int_zyx == c).astype(np.float32)
    return oh


# ============================================================
# FIX 1: Make inference prompts look like training prompts
# (training used balls radius ~3..8 and multiple clicks)
# ============================================================
def _spherical_se(radius: int) -> np.ndarray:
    zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    return (zz * zz + yy * yy + xx * xx) <= radius * radius


def dilate_prompts_per_label(prompt_int_zyx: np.ndarray, kmax: int, radius: int = 5) -> np.ndarray:
    """
    Dilate each label separately with a spherical-ish structuring element.
    This approximates your training clicks (balls) and improves propagation.
    """
    if radius <= 0:
        return prompt_int_zyx

    se = _spherical_se(int(radius))
    out = np.zeros_like(prompt_int_zyx, dtype=np.int32)

    for lbl in range(1, kmax + 1):
        m = (prompt_int_zyx == lbl)
        if not m.any():
            continue
        m_d = ndi.binary_dilation(m, structure=se)
        out[m_d] = lbl

    return out
# ============================================================


def bbox_from_mask(mask_zyx: np.ndarray, margin: int):
    zs, ys, xs = np.where(mask_zyx > 0)
    if zs.size == 0:
        return None
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(mask_zyx.shape[0], z1 + margin)
    y1 = min(mask_zyx.shape[1], y1 + margin)
    x1 = min(mask_zyx.shape[2], x1 + margin)

    return (z0, z1, y0, y1, x0, x1)


def pad_to_min_size(vol_zyx: np.ndarray, min_size=(64, 64, 64), constant=0):
    z, y, x = vol_zyx.shape
    pz = max(0, min_size[0] - z)
    py = max(0, min_size[1] - y)
    px = max(0, min_size[2] - x)

    pad = (
        (pz // 2, pz - pz // 2),
        (py // 2, py - py // 2),
        (px // 2, px - px // 2),
    )
    vol_p = np.pad(vol_zyx, pad, mode="constant", constant_values=constant)
    return vol_p, pad


def unpad(vol_zyx: np.ndarray, pad):
    (z0, z1), (y0, y1), (x0, x1) = pad
    z_slice = slice(z0, vol_zyx.shape[0] - z1 if z1 > 0 else vol_zyx.shape[0])
    y_slice = slice(y0, vol_zyx.shape[1] - y1 if y1 > 0 else vol_zyx.shape[1])
    x_slice = slice(x0, vol_zyx.shape[2] - x1 if x1 > 0 else vol_zyx.shape[2])
    return vol_zyx[z_slice, y_slice, x_slice]


@torch.inference_mode()
def run_model_sliding_window(
    model: torch.nn.Module,
    ct_zyx: np.ndarray,
    prompt_int_zyx: np.ndarray,
    device: torch.device,
    kmax: int,
):
    prompt_oh = prompt_int_to_onehot(prompt_int_zyx, kmax)
    x = np.concatenate([ct_zyx[None, ...], prompt_oh], axis=0).astype(np.float32)
    x_t = torch.from_numpy(x)[None, ...].to(device)

    logits = sliding_window_inference(
        inputs=x_t,
        roi_size=PATCH_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        predictor=model,
        overlap=OVERLAP,
        mode="gaussian",
        padding_mode="constant",
        cval=0.0,
    )  # (1, C, Z, Y, X) where C = 1+kmax

    probs = torch.softmax(logits, dim=1)[0]  # (C, Z, Y, X)

    # ------------------------------------------------------------
    # Restrict predictions to {background + prompted labels}
    # ------------------------------------------------------------
    present = np.unique(prompt_int_zyx)
    present = present[present > 0]  # only prompted label IDs
    allowed = np.concatenate([[0], present]).astype(np.int64)

    probs_np = probs.detach().cpu().numpy().astype(np.float32)  # (C,Z,Y,X)
    keep = np.zeros((kmax + 1,), dtype=bool)
    keep[allowed] = True
    probs_np[~keep, ...] = 0.0

    pred = np.argmax(probs_np, axis=0).astype(np.int32)  # (Z,Y,X)

    # enforce prompt voxels exactly (recommended)
    m = prompt_int_zyx > 0
    pred[m] = prompt_int_zyx[m]
    # ------------------------------------------------------------

    p_bg = probs_np[0]  # (Z,Y,X)
    fg_prob = (1.0 - p_bg).astype(np.float32)
    return pred, fg_prob


@torch.inference_mode()
def run_prompt_roi_inference(
    model: torch.nn.Module,
    ct_zyx: np.ndarray,
    prompt_int_zyx: np.ndarray,
    device: torch.device,
    kmax: int,
    margin: int = 24,
):
    bb = bbox_from_mask(prompt_int_zyx, margin=margin)
    if bb is None:
        pred_full = np.zeros_like(prompt_int_zyx, dtype=np.int32)
        fg_full = np.zeros_like(ct_zyx, dtype=np.float32)
        return pred_full, fg_full

    z0, z1, y0, y1, x0, x1 = bb
    ct_roi = ct_zyx[z0:z1, y0:y1, x0:x1]
    pr_roi = prompt_int_zyx[z0:z1, y0:y1, x0:x1]

    ct_roi_p, pad_ct = pad_to_min_size(ct_roi, PATCH_SIZE, constant=0.0)
    pr_roi_p, _ = pad_to_min_size(pr_roi, PATCH_SIZE, constant=0)

    pred_roi_p, fg_roi_p = run_model_sliding_window(model, ct_roi_p, pr_roi_p, device, kmax)

    pred_roi = unpad(pred_roi_p, pad_ct)
    fg_roi = unpad(fg_roi_p, pad_ct)

    pred_full = np.zeros_like(prompt_int_zyx, dtype=np.int32)
    fg_full = np.zeros_like(ct_zyx, dtype=np.float32)

    pred_full[z0:z1, y0:y1, x0:x1] = pred_roi
    fg_full[z0:z1, y0:y1, x0:x1] = fg_roi
    return pred_full, fg_full


def main():
    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
    log(f"Using device: {device}")

    if not MODEL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(MODEL_CHECKPOINT_PATH)
    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(INPUT_IMAGE_PATH)

    ct_zyx = load_nifti_as_ras_zyx(INPUT_IMAGE_PATH)
    ct_zyx = scale_intensity_minmax_like_monai(ct_zyx)
    prompt_zyx = np.zeros_like(ct_zyx, dtype=np.int32)

    model = build_model(KMAX)
    model = load_state_dict_checkpoint(model, MODEL_CHECKPOINT_PATH, device)

    viewer = napari.Viewer(title="Interactive Click-Prompt UNet (Prompt-ROI Inference)")
    viewer.add_image(ct_zyx, name="ct (RAS canonical, ZYX)", contrast_limits=(0, 1))

    prompt_layer = viewer.add_labels(prompt_zyx, name="prompt")

    pred_layer = viewer.add_labels(
        np.zeros_like(prompt_zyx, dtype=np.int32),
        name="pred",
        opacity=0.6,
    )

    prob_layer = viewer.add_image(
        np.zeros_like(ct_zyx, dtype=np.float32),
        name="fg_prob",
        opacity=0.5,
        visible=False,
    )

    viewer.layers.selection.active = prompt_layer
    prompt_layer.brush_size = 2
    prompt_layer.selected_label = 1

    @magicgui(
        call_button="Run inference (prompt ROI)",
        show_prob={"label": "Show fg_prob layer"},
        clear_prompt={"label": "Clear prompt"},
        selected_label={"label": "Selected prompt label", "min": 0, "max": KMAX},
        margin={"label": "ROI margin (vox)", "min": 0, "max": 256},
        dilate_radius={"label": "Dilate prompt radius (vox)", "min": 0, "max": 12},
    )
    def controls(
        selected_label: int = 1,
        show_prob: bool = False,
        clear_prompt: bool = False,
        margin: int = PROMPT_MARGIN,
        dilate_radius: int = DEFAULT_DILATE_RADIUS,
    ):
        show_info("Inference button clicked")
        log(">>> BUTTON CLICKED <<<")

        prompt_layer.selected_label = int(selected_label)

        if clear_prompt:
            show_info("Prompt cleared")
            prompt_layer.data[:] = 0
            pred_layer.data[:] = 0
            prob_layer.data[:] = 0
            prompt_layer.refresh()
            pred_layer.refresh()
            prob_layer.refresh()
            log(">>> PROMPT CLEARED <<<")
            return

        pr = prompt_layer.data.astype(np.int32)

        # ===== FIX 1 applied here =====
        pr_dil = dilate_prompts_per_label(pr, KMAX, radius=int(dilate_radius))
        # =============================

        # --- DEBUG ---
        u, c = np.unique(pr_dil, return_counts=True)
        log(f"PROMPT(unique, after dilation): {list(zip(u.tolist(), c.tolist()))}")
        oh = prompt_int_to_onehot(pr_dil, KMAX)
        log(f"prompt sum per channel (after dilation): {[float(oh[i].sum()) for i in range(KMAX)]}")
        # -----------

        pred, fg_prob = run_prompt_roi_inference(
            model=model,
            ct_zyx=ct_zyx,
            prompt_int_zyx=pr_dil,
            device=device,
            kmax=KMAX,
            margin=int(margin),
        )

        log(f"PRED unique: {np.unique(pred).tolist()}")

        pred_layer.data = pred
        pred_layer.refresh()

        prob_layer.data = fg_prob
        prob_layer.visible = bool(show_prob)
        prob_layer.refresh()

        show_info("Inference done. Open debug_log.txt next to inference.py")

    viewer.window.add_dock_widget(controls, area="right", name="Inference Controls")
    napari.run()


if __name__ == "__main__":
    main()