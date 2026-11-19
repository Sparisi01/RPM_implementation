# windows_manager.py>

from src.InverseHeatModel import InverseHeatModel

import torch

def make_windows(g: torch.Tensor, f: torch.Tensor, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split each sample into non-overlapping windows of size W.

    The last `n_timesteps % W` samples are discarded if
    `n_timesteps` is not divisible by `W`.

    Args:
        g (np.ndarray):
            Input array with shape (n_samples, n_timesteps).

        f (np.ndarray):
            Target array with shape (n_samples, n_timesteps).

        W (int):
            Window length in samples.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - g_wins: Array with shape (n_samples, n_windows, W).
            - f_wins: Array with shape (n_samples, n_windows, W).
    """
    
    n_samples, n_timesteps = g.shape
    n_windows = n_timesteps // W
 
    g_wins = g[:, : n_windows * W].reshape(n_samples, n_windows, W)
    f_wins = f[:, : n_windows * W].reshape(n_samples, n_windows, W)
 
    return g_wins, f_wins

def window_loss_single(model: InverseHeatModel, g_cur: torch.Tensor, g_prev: torch.Tensor, f_real: torch.Tensor, loss_fn) -> float:
    """
    Compute the mean loss over one window.

    Args:
        g_cur (np.ndarray):
            Current input windows with shape (W).

        g_prev (np.ndarray):
            Previous input windows with shape (W).

    Returns:
        float:
            Mean loss over the current window.
    """
    
    f_pred = model(g_cur, g_prev)
    return loss_fn(f_pred, f_real)

def window_loss_full(model: InverseHeatModel, g_wins: torch.Tensor, f_wins: torch.Tensor, loss_fn) -> float:
    """
    Compute the mean loss over all windows and samples.

    For each sequence of windows, the current sample is reconstructed
    using the previous `g` window as `g_prev`.

    The first window of each sample is excluded from the loss computation
    because `g_prev = 0` by definition. This assumes the system starts
    in thermal equilibrium with the environment (e.g. T = 0 for all x).

    Args:
        g_wins (np.ndarray):
            Input windows with shape (n_samples, n_windows, W).

        f_wins (np.ndarray):
            Target windows with shape (n_samples, n_windows, W).

    Returns:
        float:
            Mean loss over all samples and valid windows.
    """
    
    n_samples, n_windows, W = g_wins.shape
    device = g_wins.device
 
    preds = [] 
 
    for i_win in range(n_windows):
        g_cur = g_wins[:, i_win, :]             # (n_samples, W)
 
        if i_win == 0:
            # First windows: g_prev = zero
            g_prev = torch.zeros_like(g_cur)
        else:
            g_prev = g_wins[:, i_win - 1, :]    # (n_samples, W)
 
        f_pred = model(g_cur, g_prev)           # (n_samples, W)
        preds.append(f_pred)

    preds = torch.stack(preds, dim=1)           # (n_samples, n_windows, W)
    
    return loss_fn(preds, f_wins)