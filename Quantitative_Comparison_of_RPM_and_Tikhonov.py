from scipy import datasets

from src.RPM import LearnedTikhonov, train
from src.print_utils import print_kv, print_section
from src.Heat1DCrankNicolson import Heat1DCrankNicolson
from src.TikhonovHandler import TikhonovHandler
from src.NoiseGenerator import NoiseGenerator, print_noise_stats, apply_noise_to_dataset
from typing import Literal, get_args, Tuple
from src.LoadingHandler import load_dataset, load_impulse_response, load_simulation_info
from src.PlottingHandler import plot_diagnostics, plot_history, plot_mse_t_comparison, plot_prediction, plot_prediction_RPM
from scipy.optimize import brentq

import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("styles/plots.mplstyle")

# Training configuration
TRAINING_CONFIG = {
    "perform_training": True,
    "n_epochs": 5000,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "scheduler_type": "Cosine",
    "log_every": 100,
    "lambda_smooth": 0,                         # Smoothness regularization weight
    "mult_factor_diffusion_time": 1.5,
    "learn_phi": True,
    "learn_R": True,
    "min_omega": 0,
    "max_omega":0,
}

# Choose: "white", "correlated", "lowfreq_systematic", "multiplicative", "pink", "brown", "thermal"
DATASETS = [
    {
    "code": "DEFAULT",
    "path": "lru5_20/",
    "noise_conf": {
            "noise_level": 0,                        
            "noise_type": "white",                 
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "WHITE25",
    "path": "lru5_20_noise_25_white/",
    "noise_conf": {
            "noise_level": 0.25,                        
            "noise_type": "white",                 
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "CORR10",
    "path": "lru5_20_noise_10_correlated/",
    "noise_conf": {
            "noise_level": 0.1,
            "noise_type": "correlated",
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "CORR25",
    "path": "lru5_20_noise_25_correlated/",
    "noise_conf": {
            "noise_level": 0.25,
            "noise_type": "correlated",
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "MULT5",
    "path": "lru5_20_noise_5_mult/",
    "noise_conf": {
            "noise_level": 0.05,
            "noise_type": "multiplicative",
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "MULT20",
    "path": "lru5_20_noise_20_mult/",
    "noise_conf": {
            "noise_level": 0.20,
            "noise_type": "multiplicative",
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "LOWFREQ10",
    "path": "lru5_20_noise_10_low_freq/",
    "noise_conf": {
            "noise_level": 0.10,
            "noise_type": "lowfreq_systematic",
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "LOWFREQ25",
    "path": "lru5_20_noise_25_low_freq/",
    "noise_conf": {
            "noise_level": 0.25,
            "noise_type": "lowfreq_systematic",
            "apply_delay": False,
            "rolling": 0,
        }
    },
    {
    "code": "DELAY15",
    "path": "lru5_20_delay_15/",
    "noise_conf": {
            "noise_level": 0.05,
            "noise_type": "white",
            "apply_delay": True,
            "rolling": 15,
        }
    },
    {
    "code": "DELAY100",
    "path": "lru5_20_delay_100/",
    "noise_conf": {
            "noise_level": 0.05,
            "noise_type": "white",
            "apply_delay": True,
            "rolling": 100,
        }
    },
    {
    "code": "DELAY200",
    "path": "lru5_20_delay_200/",
    "noise_conf": {
            "noise_level": 0.05,
            "noise_type": "white",
            "apply_delay": True,
            "rolling": 200,
        }
    },
    ]


def sci_latex(x):
    mantissa, exponent = f"{x:.2e}".split("e")
    exponent = int(exponent)
    return rf"{mantissa} \times 10^{{{exponent}}}"

def main():
    
    f = open("table.txt", "w")
    
    for set in DATASETS:
        RPM_best_val,TIKH_best_val = perform_analysis(set["path"],set["noise_conf"], set["code"])
        print_kv("RPM:",sci_latex(RPM_best_val))
        print_kv("TIKHO:",sci_latex(TIKH_best_val))
        
        print(
            f'{set["code"]} & ${sci_latex(RPM_best_val)}$ & ${sci_latex(TIKH_best_val)}$ & '
            f'$\\sim {int(TIKH_best_val/RPM_best_val)} \\times$ & '
            f'\\ref{{fig:comparisonfig_{set["code"]}}}, '
            f'\\ref{{fig:loss_{set["code"]}}}, '
            f'\\ref{{fig:diagnosticsfig_{set["code"]}}}\\\\',
            file=f,
            flush=True
        )

    
    f.close()
    
def perform_analysis(DATASET_CODE,NOISE_CONFIG, code):
    
    DATASET_CODE = "dataset_sintetico_eltek/"+DATASET_CODE
    print_section(code)
    
    VAL_FRACTION = 0.5
    SEED = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ----------------------------------------------------------------------------------
    
    #print_section("LOADING DATA")

    # Load simulation parameters
    L, Tf, nt, D, h_param, l, r, dt = load_simulation_info("dataset_sintetico_eltek/")
    n = int(2 * nt - 1)

    # Load impulse response
    H_np, nk, omegas = load_impulse_response("dataset_sintetico_eltek/", n, dt)
    #print_kv("Impulse response", f"nk={nk}, n={n}")

    # Load dataset
    f_all, g_all = load_dataset("dataset_sintetico_eltek/")
    f_all = f_all[:300]
    g_all = g_all[:300]
    
    N = len(f_all)
    
    #print_kv("Dataset", f"N={N}, nt={nt}")
    
    # Noise configuration - CHOOSE YOUR SCENARIO HERE
    

    # ----------------------------------------------------------------------------------
    
    print_section("NOISE CONFIGURATION")

    noise_level = NOISE_CONFIG["noise_level"]
    noise_type = NOISE_CONFIG["noise_type"]

    print_kv("Noise type", noise_type)
    print_kv("Noise level", f"{noise_level*100:.1f}%")

    if noise_level > 0:
        generator = NoiseGenerator()
        g_all, noise = generator.apply_noise(g_all, noise_level, noise_type)
        #print_noise_stats(noise, noise_type)
    else:
        print_kv("Status", "No noise applied (clean data)")

    # Apply delay if needed
    if NOISE_CONFIG["apply_delay"] and NOISE_CONFIG["rolling"] != 0:
        rolling = NOISE_CONFIG["rolling"]
        g_rolled = np.roll(g_all, rolling, axis=1)
        g_rolled[:, :rolling] = 0
        g_all = g_rolled
        print_kv("Time delay applied", f"{rolling} samples = {rolling*dt:.4e} s")
    
    # ----------------------------------------------------------------------------------
    
    #print_section("DATASET SPLIT")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    n_val = max(1, int(VAL_FRACTION * N))
    idx_val = idx[:n_val]
    idx_train = idx[n_val:]

    print_kv("Dataset split", f"total={N}, train={len(idx_train)}, val={n_val}")
    print_kv("Device", DEVICE)

    G_train = torch.from_numpy(g_all[idx_train]).to(DEVICE)
    F_train = torch.from_numpy(f_all[idx_train]).to(DEVICE)
    G_val = torch.from_numpy(g_all[idx_val]).to(DEVICE)
    F_val = torch.from_numpy(f_all[idx_val]).to(DEVICE)

    # Initialize solver for comparison
    solver = Heat1DCrankNicolson(L=L, Tf=Tf, nt=nt, D=D, h=h_param, wanted_r=r)
    solver.info()
    th = TikhonovHandler(solver)

    # Compute and estimation for the best lambda 
    f_true = F_val.cpu().numpy()
    G_val_np = G_val.cpu().numpy()
    lambda_mean = 0
    for i in range(5):
        lambda_mean += th.best_lambda(G_val_np[i])

    nt = f_true.shape[1]
    t = np.arange(nt)

    best_lambda=lambda_mean/3
    print_kv("Best Lambda estimate:", f"{best_lambda:.3e}")
    
    # ----------------------------------------------------------------------------------
    
    #print_section("DIFFUSION TIME EVALUATION")

    def f_eigen(x, h, L):
        return x + h * np.tan(x * L)

    def trova_radici(h, L, n_radici):
        radici = []
        k = 0
        while len(radici) < n_radici:
            # Per x + h*tan(xL)=0 con h>0 le radici stanno dove tan < 0,
            # cioè in ( (2k+1)Pi/(2L) , (k+1)Pi/L )
            left_asym  = (2*k + 1) * np.pi / (2*L)   # asintoto sinistro
            right_zero = (k + 1)   * np.pi / L        # zero di tan
            
            a = left_asym  + 1e-9
            b = right_zero - 1e-9

            try:
                fa, fb = f_eigen(a, h, L), f_eigen(b, h, L)
                if fa * fb < 0:
                    root = brentq(f_eigen, a, b, args=(h, L))
                    radici.append(root)
            except Exception:
                pass
            k += 1
        return np.array(radici)

    lamb_0 = trova_radici(h_param, L, 1)[0]
    diffusion_time_0 = TRAINING_CONFIG["mult_factor_diffusion_time"]/(D*lamb_0*lamb_0)
    diffusion_time_0_inSample = int(diffusion_time_0 / dt)

    #print_kv("lambda_0:",f"{lamb_0:.6f}")
    #print_kv("diffusion_time_0:",f"{diffusion_time_0:.6f}")
    #print_kv("diffusion_time_0_inSample:",f"{diffusion_time_0_inSample}")

    min_mse_index = diffusion_time_0_inSample
    max_mse_index = -diffusion_time_0_inSample

    if NOISE_CONFIG["apply_delay"] == True:
        min_mse_index = min_mse_index + NOISE_CONFIG["rolling"]
        max_mse_index = max_mse_index - NOISE_CONFIG["rolling"]
        
    if max_mse_index == 0:
        max_mse_index = -1

    #print_kv("min_mse_index:",f"{min_mse_index}")
    #print_kv("max_mse_index:",f"{max_mse_index}")
    
    model = LearnedTikhonov(H=H_np, n=n, nt=nt, dt=dt, lambda_tik=best_lambda).to(DEVICE)

    # ----------------------------------------------------------------------------------
    
    model = LearnedTikhonov(H=H_np, n=n, nt=nt, dt=dt, lambda_tik=best_lambda).to(DEVICE)
    model.set_tainable_range_by_omegas(TRAINING_CONFIG["min_omega"], TRAINING_CONFIG["max_omega"] if TRAINING_CONFIG["max_omega"] != 0 else np.max(omegas))

    # ----------------------------------------------------------------------------------
    
    if TRAINING_CONFIG["perform_training"]:
        
        
        print_section("TRAINING PHI")
        model.unfreeze_phi()
        model.freeze_R()
        
        history, _ = train(
            model=model,
            G_train=G_train,
            F_train=F_train,
            G_val=G_val,
            F_val=F_val,
            n_epochs=TRAINING_CONFIG["n_epochs"],
            lr=TRAINING_CONFIG["lr"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            checkpoint_path=f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt",
            scheduler_type=TRAINING_CONFIG["scheduler_type"],
            log_every=TRAINING_CONFIG["log_every"],
            lambda_smooth=TRAINING_CONFIG["lambda_smooth"],
            min_mse_index=min_mse_index,
            max_mse_index=max_mse_index
        )
        
        
        print_section("TRAINING R")
        model.unfreeze_R()
        model.freeze_phi()
        
        history, _ = train(
                model=model,
                G_train=G_train,
                F_train=F_train,
                G_val=G_val,
                F_val=F_val,
                n_epochs=TRAINING_CONFIG["n_epochs"],
                lr=TRAINING_CONFIG["lr"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                checkpoint_path=f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt",
                scheduler_type=TRAINING_CONFIG["scheduler_type"],
                log_every=TRAINING_CONFIG["log_every"],
                lambda_smooth=TRAINING_CONFIG["lambda_smooth"],
                min_mse_index=min_mse_index,
                max_mse_index=max_mse_index,
                history = history
            )
        
        
        print_section("TRAINING BOTH")
        model.unfreeze_R()
        model.unfreeze_phi()
        
        history, RPM_best_val = train(
                model=model,
                G_train=G_train,
                F_train=F_train,
                G_val=G_val,
                F_val=F_val,
                n_epochs=TRAINING_CONFIG["n_epochs"]*2,
                lr=TRAINING_CONFIG["lr"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                checkpoint_path=f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt",
                scheduler_type=TRAINING_CONFIG["scheduler_type"],
                log_every=TRAINING_CONFIG["log_every"],
                lambda_smooth=TRAINING_CONFIG["lambda_smooth"],
                min_mse_index=min_mse_index,
                max_mse_index=max_mse_index,
                history = history
            )
        
    else:
        print_kv("Training", "Skipped (perform_training=False)")
        history = None
    
    # ----------------------------------------------------------------------------------
    
    print_section("DIAGNOSTICS RPM")

    # Load trained model
    ckpt = torch.load(f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt", 
                        map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to("cpu")

    # Plot diagnostics
    plot_diagnostics(model, omegas, th._build_W(best_lambda),
                    save_path=f"assets/tikhonov_learned/{DATASET_CODE}diagnostics.pdf")

    if history is not None:
        plot_history(history, 
                    save_path=f"assets/tikhonov_learned/{DATASET_CODE}loss_history.pdf")

    # Plot predictions
    RPM_best_val = plot_prediction_RPM(model, G_val.cpu(), F_val.cpu(),
                    n_samples=7,
                    save_path=f"assets/tikhonov_learned/{DATASET_CODE}predictions_RPM.pdf",
                    min_mse_index=min_mse_index,
                    max_mse_index=max_mse_index,
                    index_shift = 10)
    
    f_pred_th = np.ndarray(np.shape(f_true))
    i = 0
    for G in G_val_np:
        f_pred_th[i] = (th.regularize(G, lambda_reg=best_lambda, regularization_type_="closed"))
        i = i + 1

    model.eval()
    with torch.no_grad():
        f_pred_rpm = model(G_val.cpu()).cpu().numpy()
    
    # ----------------------------------------------------------------------------------
    
    plot_mse_t_comparison(f_true,f_pred_th,f_pred_rpm,
                    save_path=f"assets/tikhonov_learned/{DATASET_CODE}mse_t_comparison.pdf",
                    mst_path=f"assets/tikhonov_learned/{DATASET_CODE}mse_t.csv",
                    min_mse_index=min_mse_index,
                    max_mse_index=max_mse_index)
    
    TIKH_best_val = plot_prediction(G_val.cpu(),f_true,f_pred_th,n_samples=7,
                save_path=f"assets/tikhonov_learned/{DATASET_CODE}predictions_Tikh.pdf",
                min_mse_index=min_mse_index,
                max_mse_index=max_mse_index,
                index_shift = 10)
    
    return RPM_best_val,TIKH_best_val 

if __name__ == "__main__":
    main()