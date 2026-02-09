import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_history(path: str):
    with open(path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    return hist


def main():
    out_dir = "./runs_clickprompts"
    history_path = os.path.join(out_dir, "history.json")

    hist = load_history(history_path)

    epochs = np.array(hist["epoch"], dtype=np.int32)
    train_loss = np.array(hist["train_loss"], dtype=np.float32)

    # val may contain None → convert to NaN
    val = [(np.nan if v is None else float(v))
           for v in hist.get("val_fg_union_dice", [])]
    val_fg_union_dice = np.array(val, dtype=np.float32)

    # -------------------------
    # Plot: loss + val dice
    # -------------------------
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val FG Union Dice")
    ax2.plot(epochs, val_fg_union_dice, linestyle="--", label="Val FG Dice")

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training Loss & Validation FG Union Dice")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
