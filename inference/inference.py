import numpy as np
import torch
import napari
from pathlib import Path
from magicgui import magicgui
import nibabel as nib

from monai.networks.nets import UNet


# ============================================================
# CONFIG
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


# ----------------------------
# MODEL
# ----------------------------
def build_model(kmax: int):
    return UNet(
        spatial_dims=3,
        in_channels=1 + kmax,
        out_channels=1 + kmax,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    )


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(str(ckpt_path), map_location=device)

    # If nested formats appear, unwrap them
    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "network"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    # 🔑 Your checkpoint has keys starting with "model."
    # The UNet instance expects keys without that prefix.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval()
    return model


# ----------------------------
# DATA
# ----------------------------
def load_ct(path: Path):
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = np.moveaxis(vol, -1, 0)  # -> (D,H,W)
    vmin, vmax = np.percentile(vol, (1, 99))
    vol = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return vol


def prompt_to_onehot(prompt, kmax):
    oh = np.zeros((kmax,) + prompt.shape, dtype=np.float32)
    for c in range(1, kmax + 1):
        oh[c - 1] = (prompt == c)
    return oh


@torch.inference_mode()
def infer(model, ct, prompt, device, kmax):
    x = np.concatenate([ct[None], prompt_to_onehot(prompt, kmax)], axis=0)
    x = torch.from_numpy(x[None]).to(device)
    logits = model(x)
    return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int32)


# ----------------------------
# NAPARI
# ----------------------------
def main():
    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    model = build_model(KMAX)
    model = load_checkpoint(model, MODEL_CHECKPOINT_PATH, device)

    ct = load_ct(INPUT_IMAGE_PATH)
    prompt = np.zeros_like(ct, dtype=np.int32)

    # 🔑 THIS LINE IS THE KEY
    viewer = napari.Viewer(title="3D CT + Interactive Segmentation", ndisplay=3)

    ct_layer = viewer.add_image(ct, name="ct")
    ct_layer.rendering = "attenuated_mip"
    ct_layer.contrast_limits = (0, 1)

    prompt_layer = viewer.add_labels(prompt, name="prompt")
    pred_layer = viewer.add_labels(prompt.copy(), name="pred", opacity=0.6)

    prompt_layer.brush_size = 2
    prompt_layer.selected_label = 1
    viewer.layers.selection.active = prompt_layer

    @magicgui(
        call_button="Run inference",
        label={"label": "Paint label", "min": 0, "max": KMAX},
    )
    def run(label=1):
        prompt_layer.selected_label = label
        pred = infer(model, ct, prompt_layer.data, device, KMAX)
        pred_layer.data = pred
        pred_layer.refresh()

    viewer.window.add_dock_widget(run, area="right")

    print("\n✔ Use the 2D / 3D toggle (bottom-left) to switch views.")
    print("✔ Rotate in 3D with mouse.")
    print("✔ Paint prompts (1..8) then click Run inference.\n")

    napari.run()


if __name__ == "__main__":
    main()
