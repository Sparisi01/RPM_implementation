from src.LearnedTikhonov import LearnedTikhonov
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.print_utils import print_kv, print_section

def compute_mse_t(f_pred, f_true):
    return np.mean((f_pred - f_true) ** 2, axis=0)

def plot_diagnostics(model: LearnedTikhonov, omegas: np.ndarray, tikh_w: np.ndarray = None,
                    save_path: str = "diagnostics.pdf"):
    """Plot learned R, Phi, and comparison with Tikhonov."""
    fig, axes = plt.subplots(3, 1, figsize=(3.2, 3.6), sharex=True)
    ax1, ax2, ax3 = axes
    
    ax1.plot(omegas, model.get_phi(), color="steelblue", lw=1.5)
    ax1.set_ylabel(r"$\phi$")
    ax1.set_title(r"Learned model")
    ax1.set_ylim([-3.3, 3.3])
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(omegas, model.get_R(), color="darkorange", lw=1)
    ax2.set_ylabel(r"$R$")
    ax2.grid(True, alpha=0.3)

    ax3.semilogy(omegas, model.get_W_magnitude(), color="r", lw=1, label=r"Learned $|\Lambda|$")
    if tikh_w is not None:
        ax3.semilogy(omegas, np.abs(tikh_w), color="b", lw=1, alpha=1,ls="--", label=r"Tikhonov $|\Lambda|$")
    
    ax3.set_ylabel(r"$|\Lambda|$")
    ax3.set_xlabel(r"$\omega$")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Diagnostic saved", f"{save_path}")


def plot_history(history: dict, save_path: str = "loss_history.pdf"):
    """Plot training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots()
    
    ax.semilogy(epochs, history["train_loss"], label="Train MSE")
    ax.semilogy(epochs, history["val_loss"], label="Val MSE", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Loss history saved", f"{save_path}")


def plot_prediction_RPM(model: LearnedTikhonov,
                   G_val: torch.Tensor,
                   F_val: torch.Tensor,
                   n_samples: int = 3,
                   save_path: str = "predictions.pdf",
                   min_mse_index: int = 0,
                   max_mse_index: int = -1):
    model.eval()
    with torch.no_grad():
        f_pred = model(G_val).cpu().numpy()
    f_true = F_val.cpu().numpy()
    
    plot_prediction(G_val,f_true,f_pred,model_name="RPM",n_samples=n_samples, save_path=save_path, min_mse_index=min_mse_index, max_mse_index=max_mse_index)
    
def plot_prediction(G_val: torch.Tensor,
                   f_true,
                   f_pred,
                   n_samples: int = 3,
                   model_name: str = "Tikhonov",
                   save_path: str = "predictions.pdf",
                   min_mse_index: int = 0,
                   max_mse_index: int = -1):
    nt = f_true.shape[1]
    t = np.arange(nt)

    n_show = min(n_samples, len(f_true))
    fig, axes = plt.subplots(n_show, 1, figsize=(3.2, 1.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    G_array = G_val.cpu().numpy()
    mse_array = np.mean((f_pred[:,min_mse_index:max_mse_index] - f_true[:,min_mse_index:max_mse_index]) ** 2, axis=1)
    
    for i, ax in enumerate(axes):
        ax.plot(t, G_array[i], label=r"$g_{obs}$",ls="-.", c="k")
        ax.plot(t, f_true[i], label=r"$f_{true}$", lw=3, c="b", alpha=0.5)
        ax.plot(t, f_pred[i], label=r"$f_{rec}$",ls="--", c="r")
        
        mse_i = mse_array[i]
        ax.text(0.12, 0.15, f"MSE = {mse_i:.3e}",
                transform=ax.transAxes, verticalalignment='top')
        
        ax.set_ylabel(r"$\Theta$")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 5])
        
        if min_mse_index > 0:
            ax.axvline(x=min_mse_index, ls="--", c="k", alpha=0.5)
        if max_mse_index < -1:
            ax.axvline(x=nt+max_mse_index, ls="--", c="k", alpha=0.5)
        
        ax.minorticks_on()

    valMSE = np.mean(mse_array)
    axes[0].legend(loc="upper center", ncols=3)
    axes[-1].set_xlabel("Sample")
    axes[0].set_title(model_name + f" - MSE: {valMSE:.3e}")
    
    print_kv("Val MSE (RPM):", f"{valMSE:.3e}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Predictions saved", f"{save_path}")
    return valMSE
    
def plot_mse_t_comparison(
                f_true,
                f_pred_th,
                f_pred_rpm,
                mst_path: str = "",
                save_path: str = "mse_t_comparison.pdf",
                min_mse_index: int = 0,
                max_mse_index: int = -1):

    nt = f_true.shape[1]
    t = np.arange(nt)
        
    mse_t_th = compute_mse_t(f_pred_th,f_true)
    mse_t_rpm = compute_mse_t(f_pred_rpm,f_true)

    plt.ylabel(r"$MSE_t$")
    plt.xlabel(r"Sample")
    plt.plot(t,mse_t_th, label=r"Tikhonov")
    plt.plot(t,mse_t_rpm, label=r"RPM")
    plt.legend()
    plt.grid()
    plt.title(r"$MSE_t$ comparison between RPM and Tikhonov")
    plt.yscale("log")
    
    if min_mse_index > 0:
        plt.axvline(x=min_mse_index, ls="--", c="k", alpha=0.5)
    if max_mse_index < -1:
        plt.axvline(x=nt+max_mse_index, ls="--", c="k", alpha=0.5)
            
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print_kv(f"Predictions saved", f"{save_path}")
    
    if mst_path != "":
        np.savetxt(mst_path, (mse_t_rpm,mse_t_th), delimiter=';')