import numpy as np
import torch
import napari
from pathlib import Path
from magicgui import magicgui
import nibabel as nib

from monai.networks.nets import UNet


# ============================================================
# CONFIG – EDIT THESE TWO PATHS
# ============================================================
MODEL_CHECKPOINT_PATH = Path(
    r"C:/uniDev/forschungsprojekt/trainer/runs_clickprompts/model_best.pth"
)

INPUT_IMAGE_PATH = Path(
    r"C:/uniDev/forschungsprojekt/trainingsdata/nnUNet_raw/Dataset001_CADSynthetic/imagesTr/case_0001_0000.nii.gz"
)

KMAX = 8
FORCE_CPU = False
# ============================================================


def build_model(kmax: int = 8):
    model = UNet(
        spatial_dims=3,
        in_channels=1 + kmax,
        out_channels=1 + kmax,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    # 🔑 IMPORTANT: unwrap MONAI internal "model" container
    if hasattr(model, "model"):
        model = model.model

    return model


def load_state_dict_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    state = torch.load(str(ckpt_path), map_location=device)

    # allow nested dict formats too (just in case)
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    # strip common prefixes
    cleaned = {}
    for k, v in state.items():
        for prefix in ["model.", "net.", "network."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if missing:
        print("[WARN] Missing keys (showing up to 10):", missing[:10])
    if unexpected:
        print("[WARN] Unexpected keys (showing up to 10):", unexpected[:10])

    model.to(device).eval()
    return model


def load_nifti_volume(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    vol = nii.get_fdata().astype(np.float32)  # usually (X,Y,Z) or (Z,Y,X) depending on file
    # We won't reorient here—training did Orientationd("RAS") in MONAI,
    # but for inference visualization this is OK as long as you're consistent.
    return vol


def normalize_like_training(vol: np.ndarray) -> np.ndarray:
    """
    Training used MONAI ScaleIntensityd on 'image'.
    For inference we apply a reasonable equivalent:
      - robust min/max (1..99 percentile)
      - scale to [0,1]
    """
    vmin, vmax = np.percentile(vol, (1, 99))
    vol = (vol - vmin) / (vmax - vmin + 1e-8)
    return np.clip(vol, 0.0, 1.0).astype(np.float32)


def prompt_int_to_onehot(prompt_int: np.ndarray, kmax: int) -> np.ndarray:
    """
    prompt_int: (D,H,W) int in [0..kmax]
    returns: (kmax, D,H,W) float32 onehot for labels 1..kmax
    """
    oh = np.zeros((kmax,) + prompt_int.shape, dtype=np.float32)
    for c in range(1, kmax + 1):
        oh[c - 1] = (prompt_int == c).astype(np.float32)
    return oh


@torch.inference_mode()
def run_model(model: torch.nn.Module, ct_vol: np.ndarray, prompt_int: np.ndarray, device: torch.device, kmax: int):
    """
    ct_vol:     (D,H,W) float32 in [0,1]
    prompt_int: (D,H,W) int in [0..kmax]
    returns:
      pred_lbl: (D,H,W) int in [0..kmax]
      fg_prob:  (D,H,W) float32 = 1 - p(background)
    """
    prompt_oh = prompt_int_to_onehot(prompt_int, kmax)  # (kmax,D,H,W)

    # to torch: (1, C, D, H, W)
    x = np.concatenate([ct_vol[None, ...], prompt_oh], axis=0).astype(np.float32)  # (1+kmax,D,H,W)
    x_t = torch.from_numpy(x)[None, ...].to(device)

    logits = model(x_t)  # (1, 1+kmax, D, H, W)
    probs = torch.softmax(logits, dim=1)[0]  # (1+kmax, D,H,W)

    pred = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.int32)  # (D,H,W)

    p_bg = probs[0].detach().cpu().numpy().astype(np.float32)
    fg_prob = (1.0 - p_bg).astype(np.float32)

    return pred, fg_prob


def main():
    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    if not MODEL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(MODEL_CHECKPOINT_PATH)
    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(INPUT_IMAGE_PATH)

    # Load data
    ct = load_nifti_volume(INPUT_IMAGE_PATH)
    ct = normalize_like_training(ct)

    # Ensure it's (D,H,W) for the network input.
    # Many NIfTIs load as (H,W,D) or (X,Y,Z). We will convert to (D,H,W) by moving last axis to first.
    # If your data already is (D,H,W), this still works if D is the last axis.
    if ct.shape[0] != ct.shape[-1]:
        ct_dhw = np.moveaxis(ct, -1, 0)
    else:
        ct_dhw = ct

    # Init empty prompts
    prompt = np.zeros_like(ct_dhw, dtype=np.int32)

    # Model
    model = build_model(KMAX)
    model = load_state_dict_checkpoint(model, MODEL_CHECKPOINT_PATH, device)

    # Napari UI
    viewer = napari.Viewer(title="Interactive Click-Prompt UNet (MONAI)")

    viewer.add_image(ct_dhw, name="ct", contrast_limits=(0, 1))

    # Prompt layer: paint integers 0..KMAX
    prompt_layer = viewer.add_labels(
    prompt,
    name="prompt",
)

    pred_layer = viewer.add_labels(
        np.zeros_like(prompt, dtype=np.int32),
        name="pred",
        opacity=0.6,
    )

    # Foreground probability
    prob_layer = viewer.add_image(
        np.zeros_like(ct_dhw, dtype=np.float32),
        name="fg_prob",
        opacity=0.5,
        visible=False,
    )

    # Helpful defaults
    viewer.layers.selection.active = prompt_layer
    prompt_layer.brush_size = 2  # adjust to your liking
    prompt_layer.selected_label = 1  # start painting label 1

    @magicgui(
        call_button="Run inference",
        show_prob={"label": "Show fg_prob layer"},
        clear_prompt={"label": "Clear prompt"},
        selected_label={"label": "Selected prompt label", "min": 0, "max": KMAX},
    )
    def controls(
        selected_label: int = 1,
        show_prob: bool = False,
        clear_prompt: bool = False,
    ):
        # allow changing selected label from widget
        prompt_layer.selected_label = int(selected_label)

        if clear_prompt:
            prompt_layer.data[:] = 0
            pred_layer.data[:] = 0
            prob_layer.data[:] = 0
            prompt_layer.refresh()
            pred_layer.refresh()
            prob_layer.refresh()
            return

        pred, fg_prob = run_model(model, ct_dhw, prompt_layer.data.astype(np.int32), device, KMAX)

        pred_layer.data = pred
        pred_layer.refresh()

        prob_layer.data = fg_prob
        prob_layer.visible = bool(show_prob)
        prob_layer.refresh()

    viewer.window.add_dock_widget(controls, area="right")
    napari.run()


if __name__ == "__main__":
    main()
