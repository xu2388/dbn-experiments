"""Generate report figures for the DBN experiment (run AFTER training):
1. accuracy / loss curves comparing: no-pretrain vs pretrain vs pretrain+binarization
2. digit reconstruction visualization (original vs reconstructed) for digits 0-9
Outputs are saved under ./results_dbn/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, "results_dbn")
os.makedirs(OUT, exist_ok=True)

CSV = {
    "no-pretrain": "DBN_without_pretraining_classifier.csv",
    "pretrain": "DBN_with_pretraining_classifier.csv",
    "pretrain+bin": "DBN_with_pretraining_and_input_binarization_classifier.csv",
}
COLORS = {"no-pretrain": "#c0392b", "pretrain": "#2a5fd0", "pretrain+bin": "#27ae60"}


def plot_acc_loss():
    frames = {k: pd.read_csv(os.path.join(PROJ, v)) for k, v in CSV.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, df in frames.items():
        axes[0].plot(df["epochs"], df["train loss"], "o-", color=COLORS[name], alpha=0.85, label=f"{name} train")
        axes[0].plot(df["epochs"], df["test loss"], "s--", color=COLORS[name], alpha=0.6, label=f"{name} test")
        axes[1].plot(df["epochs"], df["train acc"], "o-", color=COLORS[name], alpha=0.85, label=f"{name} train")
        axes[1].plot(df["epochs"], df["test acc"], "s--", color=COLORS[name], alpha=0.6, label=f"{name} test")
    axes[0].set_title("MNIST DBN - loss (cross-entropy)")
    axes[0].set_xlabel("epochs"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title("MNIST DBN - accuracy")
    axes[1].set_xlabel("epochs"); axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout()
    fp = os.path.join(OUT, "acc_loss_curves.png")
    plt.savefig(fp, dpi=150)
    plt.close(fig)
    print("Saved", fp)
    return frames


def reconstruct_digits():
    from load_dataset import MNIST
    from DBN import DBN

    mnist = MNIST()
    tx, ty, test_x, test_y = mnist.load_dataset()

    dbn = DBN(tx.shape[1], [512, 128, 64, 10], gpu=False)
    dbn.layer_parameters = torch.load(os.path.join(PROJ, "mnist_trained_dbn.pt"), map_location="cpu", weights_only=False)

    test_y = test_y.long()
    digits = sorted(test_y.unique().tolist())
    n = len(digits)
    cols, rows = 2, n
    fig, axes = plt.subplots(rows, cols, figsize=(4.4, 2.2 * n))
    axes = np.atleast_2d(axes)

    for i, d in enumerate(digits):
        idx = (test_y == d).nonzero()[0].item()
        x = test_x[idx].to(torch.float).view(1, -1)
        recon, _ = dbn.reconstructor(x.to(torch.float))
        orig_img = x.view(28, 28).cpu().numpy()
        recon_img = recon.view(28, 28).detach().cpu().numpy()
        for j, (img, title) in enumerate([(orig_img, f"original {d}"), (recon_img, f"reconstructed {d}")]):
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].set_title(title, fontsize=9)
            axes[i, j].axis("off")

    plt.tight_layout()
    fp = os.path.join(OUT, "digit_reconstruction.png")
    plt.savefig(fp, dpi=150)
    plt.close(fig)
    print("Saved", fp)


if __name__ == "__main__":
    frames = plot_acc_loss()
    reconstruct_digits()