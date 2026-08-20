# 📘 Guida di Integrazione - Notebook Riordinato

## Riepilogo delle Modifiche

Il tuo notebook è stato **riordinato e migliorato** con:

1. ✅ **Struttura logica chiara** (7 sezioni ben definite)
2. ✅ **Libreria completa di rumore** (4 scenari realistici)
3. ✅ **Configurazione centralizzata** (parametri in un unico dizionario)
4. ✅ **Funzioni helper** (training, plotting riorganizzate)
5. ✅ **Stima della correlazione D** (calcolo automatico nelle statistiche del rumore)

---

## 🚀 Come Utilizzare

### Opzione 1: Usa il Script Python Direttamente

Se preferisci lavorare da **command line**:

```bash
# Copia il file nella tua directory
cp notebook_improved.py my_thesis/

# Esegui
cd my_thesis
python notebook_improved.py
```

Modifica i parametri in cima al file:

```python
# Notebook_improved.py, righe ~145-165

NOISE_CONFIG = {
    "noise_level": 0.05,  # Cambio: 0.05 = 5%
    "noise_type": "correlated",  # Cambio qui: "white", "correlated", "lowfreq_systematic", "multiplicative"
    "apply_delay": False,
    "rolling": 0,
}

TRAINING_CONFIG = {
    "perform_training": True,  # Cambio qui per iniziare l'addestramento
    "n_epochs": 50000,
    "lr": 1e-3,
    ...
}
```

---

### Opzione 2: Integra nel Tuo Jupyter Notebook (CONSIGLIATO)

Segui questa **struttura in celle**:

#### **Cella 1: Imports & Setup**

```python
from src.LearnedTikhonov import LearnedTikhonov
from src.print_utils import print_kv, print_section
from src.Heat1DCrankNicolson import Heat1DCrankNicolson
from src.TikhonovHandler import TikhonovHandler

import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Tuple
from scipy.ndimage import gaussian_filter1d

plt.style.use("styles/plots.mplstyle")
```

#### **Cella 2: Noise Generator Class**

Copia **intera** la classe `NoiseGenerator` da notebook_improved.py (righe ~27-110).

Oppure più semplicemente:

```python
# Importa dal file
import sys
sys.path.insert(0, '/path/to/noise_generators.py')
from noise_generators import NoiseGenerator, print_noise_stats
```

#### **Cella 3: Utility Functions (Training)**

Copia le funzioni:
- `smoothness_regularization()`
- `train()`

(Righe ~115-215 di notebook_improved.py)

#### **Cella 4: Utility Functions (Plotting)**

Copia:
- `plot_diagnostics()`
- `plot_history()`
- `plot_prediction()`

#### **Cella 5: Configuration**

```python
# Dataset configuration
DATASET_CODE = "lru5_20_D0_1/"
VAL_FRACTION = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== NOISE CONFIGURATION - MODIFICA QUI =====
NOISE_CONFIG = {
    "noise_level": 0.05,           # 5% di rumore
    "noise_type": "white",         # Scelta: "white", "correlated", "lowfreq_systematic", "multiplicative"
    "apply_delay": False,
    "rolling": 0,
}

# Training configuration
TRAINING_CONFIG = {
    "perform_training": False,
    "n_epochs": 50000,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "scheduler_type": "Cosine",
    "log_every": 100,
    "lambda_smooth": 0,
}

print_kv("Configuration", f"Noise: {NOISE_CONFIG['noise_type']}, Level: {NOISE_CONFIG['noise_level']*100:.1f}%")
```

#### **Cella 6: Data Loading**

```python
# Load simulation info
L, Tf, nt, D, h_param, l, _, r, _ = np.loadtxt(
    f"dataset/{DATASET_CODE}simulation_info.csv", delimiter=";", usecols=1
)
nt = int(nt)
dt = Tf / nt
n = int(2 * nt - 1)

print_kv("Simulation", f"nt={nt}, n={n}, dt={dt:.4e}")

# Load impulse response
h_vals = np.loadtxt(f"dataset/{DATASET_CODE}impulse_response.csv", delimiter=";")
H_np = np.fft.rfft(h_vals, n).astype(np.complex64)
nk = len(H_np)
freqs = np.fft.rfftfreq(n, d=dt)
omegas = freqs * 2 * np.pi

# Load dataset
H5_PATH = f"dataset/{DATASET_CODE}dataset.h5"
with h5py.File(H5_PATH, "r") as hf:
    data = hf["dataset"][:]

f_all = data[:, 0, :].astype(np.float32)
g_all = data[:, 1, :].astype(np.float32)

print_kv("Dataset", f"N={len(f_all)}, nt={nt}")
```

#### **Cella 7: NOISE APPLICATION (LA PARTE NUOVA!)**

```python
print_section("NOISE APPLICATION")

# Configuration
noise_level = NOISE_CONFIG["noise_level"]
noise_type = NOISE_CONFIG["noise_type"]

print_kv("Noise type", noise_type)
print_kv("Noise level", f"{noise_level*100:.1f}%")

# Apply noise
if noise_level > 0:
    generator = NoiseGenerator(seed=SEED)
    g_all, noise = generator.apply_noise(g_all, noise_level, noise_type)
    
    # Print statistics
    print(f"\n  Scenario: {noise_type}")
    print(f"    Mean:      {np.mean(noise):12.4e}")
    print(f"    Std:       {np.std(noise):12.4e}")
    print(f"    Max:       {np.max(np.abs(noise)):12.4e}")
    
    # BONUS: Stima D (correlazione noise-segnale)
    noise_freq = np.fft.rfft(noise, axis=1)
    f_freq = np.fft.rfft(f_all, axis=1)
    D_empirical = np.mean([noise_freq[i] * np.conj(f_freq[i]) 
                           for i in range(len(f_all))], axis=0)
    
    ratio_D_C = np.abs(D_empirical) / (np.abs(H_np * np.fft.rfft(f_all[0])) + 1e-10)
    print(f"    D/C ratio: {np.mean(ratio_D_C):.4f} ← Indica correlazione!")
    
    if np.mean(ratio_D_C) < 0.1:
        print(f"    ✓ Rumore quasi indipendente (D ≈ 0)")
    else:
        print(f"    ⚠ Rumore correlato (D ≠ 0) - RPM deve includere fase!")
else:
    print_kv("Status", "No noise applied (clean data)")

# Apply delay if needed
if NOISE_CONFIG["apply_delay"] and NOISE_CONFIG["rolling"] != 0:
    rolling = NOISE_CONFIG["rolling"]
    g_rolled = np.roll(g_all, rolling, axis=1)
    g_rolled[:, :rolling] = 0
    g_all = g_rolled
    print_kv("Time delay", f"{rolling} samples = {rolling*dt:.4e} s")
```

#### **Cella 8: Dataset Split**

```python
# Dataset split
N = len(f_all)
rng = np.random.default_rng(SEED)
idx = rng.permutation(N)
n_val = max(1, int(VAL_FRACTION * N))
idx_val = idx[:n_val]
idx_train = idx[n_val:]

print_kv("Dataset", f"total={N}, train={len(idx_train)}, val={n_val}")
print_kv("Device", DEVICE)

G_train = torch.from_numpy(g_all[idx_train]).to(DEVICE)
F_train = torch.from_numpy(f_all[idx_train]).to(DEVICE)
G_val = torch.from_numpy(g_all[idx_val]).to(DEVICE)
F_val = torch.from_numpy(f_all[idx_val]).to(DEVICE)
```

#### **Cella 9: Model Init**

```python
# Init model
model = LearnedTikhonov(H=H_np, n=n, nt=nt, dt=dt).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print_kv("Model parameters", f"phi: {nk}, log_R: {nk} (total: {n_params})")
```

#### **Cella 10: Training**

```python
perform_training = TRAINING_CONFIG["perform_training"]
history = None

if perform_training:
    print_section("Training")
    
    history = train(
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
    )
else:
    print_kv("Training", "Skipped")
```

#### **Cella 11: Diagnostics**

```python
print_section("Diagnostics")

# Load trained model
ckpt = torch.load(f"assets/tikhonov_learned/{DATASET_CODE}best_tikhonov.pt", 
                  map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.to("cpu")

# Solver
solver = Heat1DCrankNicolson(L=L, Tf=Tf, nt=nt, D=D, h=h_param, wanted_r=r)
th = TikhonovHandler(solver)

# Plots
plot_diagnostics(model, omegas, th._build_W(),
                save_path=f"assets/tikhonov_learned/{DATASET_CODE}diagnostics.pdf")

if history is not None:
    plot_history(history, 
                save_path=f"assets/tikhonov_learned/{DATASET_CODE}loss_history.pdf")

plot_prediction(model, G_val.cpu(), F_val.cpu(), dt,
               n_samples=7,
               save_path=f"assets/tikhonov_learned/{DATASET_CODE}predictions.pdf")

print_section("✓ Complete!")
```

---

## 🔄 Come Testare i 4 Scenari di Rumore

Esegui queste **4 celle consecutive** (copia la cella 7 e 10-11 tre volte, cambio solo il rumore):

### Test 1: Rumore Bianco (D ≈ 0)
```python
NOISE_CONFIG["noise_type"] = "white"
NOISE_CONFIG["noise_level"] = 0.05
# Esegui cella 7 + 8-11
```

### Test 2: Rumore Correlato (D ≠ 0)
```python
NOISE_CONFIG["noise_type"] = "correlated"
NOISE_CONFIG["noise_level"] = 0.05
# Esegui cella 7 + 8-11
```

### Test 3: Rumore Sistematico (D a basse freq)
```python
NOISE_CONFIG["noise_type"] = "lowfreq_systematic"
NOISE_CONFIG["noise_level"] = 0.05
# Esegui cella 7 + 8-11
```

### Test 4: Rumore Moltiplicativo (D proporzionale a g)
```python
NOISE_CONFIG["noise_type"] = "multiplicative"
NOISE_CONFIG["noise_level"] = 0.05
# Esegui cella 7 + 8-11
```

---

## 📊 Output Generati

Per ogni test otterrai:

```
assets/tikhonov_learned/lru5_20_D0_1/
├── best_tikhonov.pt          ← Modello addestrato
├── diagnostics.pdf           ← R(ω), Φ(ω), comparazione
├── loss_history.pdf          ← Curva di training
├── predictions.pdf           ← Ricostruzioni vs ground truth
└── mse.pdf                   ← MSE temporale
```

---

## 🎯 Cosa Scrivere nella Tesi

Per ogni scenario:

```
Test Scenario: [white/correlated/lowfreq/multiplicative]
Noise level: 5%
D/C ratio: [valore stimato]
Val MSE: [valore ottenuto]

Interpretazione:
- Se D/C < 0.1: Rumore indipendente, φ* ≈ 0
- Se D/C > 0.3: Rumore correlato, φ* ≠ 0 (importante!)
- Miglioramento MSE con D incluso: [percentuale]%
```

---

## ✨ Bonus: Aggiungere Più Metriche

Nella cella 7, dopo il calcolo di D, aggiungi:

```python
# Stima fase ottimale
phase_factor = np.angle(D_empirical) if np.mean(np.abs(D_empirical)) > 1e-3 else 0
print(f"    φ* (optimal phase): {phase_factor:.4f} rad")

# Stima regolarizzazione ottimale (semplificata)
SNR = np.mean(np.std(g_all, axis=1)**2) / np.mean(np.std(noise, axis=1)**2)
print(f"    SNR: {10*np.log10(SNR):.2f} dB")
```

---

## 🐛 Troubleshooting

**Errore: "NoiseGenerator not found"**
- Assicurati di aver copiato la classe nella cella 2, oppure importala correttamente

**Errore: "Device cuda not available"**
- Cambio automatico a CPU, è OK

**Errore: "File not found in dataset/"**
- Verifica il percorso `DATASET_CODE` corrisponde al tuo dataset

---

**Domande? Guarda i file scaricati:**
- `noise_generators.py` - Libreria standalone
- `notebook_improved.py` - Versione completa  
- `test_noise_correlation.py` - Script di validazione
