
# Comparação de métodos de limiarização em documentos manuscritos

""""
Métodos (scikit-image):
  1. Global fixo   -> limiar escolhido a priori (GLOBAL_T)
  2. Otsu          -> limiar global automático (threshold_otsu)
  3. Local         -> média gaussiana da vizinhança (threshold_local)
  4. Sauvola       -> local, padrão na literatura de documentos (opcional)
"""

from pathlib import Path
import csv
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import correlate
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, threshold_local, threshold_sauvola

# Configuração
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "original"
GT_DIR = DATA_DIR / "gt"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

EXTS = {".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}

GLOBAL_T = 0.5           
LOCAL_BLOCK_SIZE = 35    
LOCAL_METHOD = "gaussian"
LOCAL_OFFSET = 0.04      
INCLUDE_SAUVOLA = True
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.2
HIST_BINS = 256

SUFFIX_RE = re.compile(r"_(gt|estgt|in|bin|original)$", re.IGNORECASE)


# Leitura e pareamento
def base_name(stem: str) -> str:
    return SUFFIX_RE.sub("", stem)


def list_images(folder: Path) -> dict:
    return {base_name(p.stem): p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS}


def pair_files(img_dir: Path, gt_dir: Path) -> list:
    imgs, gts = list_images(img_dir), list_images(gt_dir)
    common = sorted(set(imgs) & set(gts), key=lambda s: (len(s), s))
    missing = set(imgs) ^ set(gts)
    if missing:
        warnings.warn(f"Arquivos sem par (ignorados): {sorted(missing)}")
    if not common:
        raise ValueError("Nenhum par original/GT encontrado. Verifique os nomes dos arquivos.")
    return [(name, imgs[name], gts[name]) for name in common]


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3] if img.shape[-1] >= 3 else img[..., 0]
        img = rgb2gray(img) if img.ndim == 3 else img
    return img_as_float(img)


def load_image(path: Path) -> np.ndarray:
    return to_gray(io.imread(path))


def load_gt(path: Path) -> np.ndarray:
    """Retorna máscara booleana: True = tinta (texto)."""
    return to_gray(io.imread(path)) < 0.5




# Limiarização (convenção: True = tinta, ou seja, pixel ESCURO)
def binarize(gray: np.ndarray) -> dict:
    t_otsu = threshold_otsu(gray)
    t_local = threshold_local(gray, block_size=LOCAL_BLOCK_SIZE,
                              method=LOCAL_METHOD, offset=LOCAL_OFFSET)
    methods = {
        "Global": {"t": GLOBAL_T, "mask": gray < GLOBAL_T},
        "Otsu": {"t": t_otsu, "mask": gray < t_otsu},
        "Local": {"t": t_local, "mask": gray < t_local},
    }
    if INCLUDE_SAUVOLA:
        t_sau = threshold_sauvola(gray, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K)
        methods["Sauvola"] = {"t": t_sau, "mask": gray < t_sau}
    return methods


# Métricas (DIBCO). Entradas: máscaras booleanas, True = tinta.
def _drd_weights(n: int = 5) -> np.ndarray:
    c = n // 2
    i, j = np.mgrid[:n, :n]
    d = np.hypot(i - c, j - c)
    w = np.zeros_like(d)
    w[d > 0] = 1.0 / d[d > 0]
    return w / w.sum()


DRD_W = _drd_weights(5)


def drd(pred: np.ndarray, gt: np.ndarray, block: int = 8) -> float:
    """Distance Reciprocal Distortion (Lu et al., 2004), como usado no DIBCO."""
    g = gt.astype(np.float64)
    # Soma ponderada do GT na vizinhança 5x5 de cada pixel
    s = correlate(g, DRD_W, mode="constant", cval=0.0)
    flipped = pred != gt
    contrib = np.where(pred, 1.0 - s, s)
    total = contrib[flipped].sum()

    h, w = gt.shape
    H, W = int(np.ceil(h / block)) * block, int(np.ceil(w / block)) * block
    padded = np.zeros((H, W), dtype=np.int32)
    padded[:h, :w] = gt
    sums = padded.reshape(H // block, block, W // block, block).sum(axis=(1, 3))
    nubn = np.count_nonzero((sums > 0) & (sums < block * block))
    return total / nubn if nubn else np.nan


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fm = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mse = np.mean(pred != gt)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else np.inf
    return {
        "FM": 100 * fm,
        "Precisao": 100 * precision,
        "Revocacao": 100 * recall,
        "PSNR": psnr,
        "DRD": drd(pred, gt),
    }


# Processamento
pairs = pair_files(IMG_DIR, GT_DIR)
print(f"{len(pairs)} pares encontrados: {[p[0] for p in pairs]}")

results = []
for name, img_path, gt_path in pairs:
    gray = load_image(img_path)
    gt = load_gt(gt_path)
    if gray.shape != gt.shape:
        warnings.warn(f"{name}: imagem {gray.shape} e GT {gt.shape} com tamanhos diferentes; ignorada.")
        continue
    methods = binarize(gray)
    for m in methods.values():
        m["metrics"] = evaluate(m["mask"], gt)
    results.append({"name": name, "gray": gray, "gt": gt, "methods": methods})

METHOD_NAMES = list(results[0]["methods"].keys())
COLORS = {"Global": "tab:red", "Otsu": "tab:blue", "Local": "tab:green", "Sauvola": "tab:purple"}


# 1) Histogramas com os limiares
def plot_histogram(ax, r, legend=True):
    ax.hist(r["gray"].ravel(), bins=HIST_BINS, range=(0, 1), color="0.6")
    for key, m in r["methods"].items():
        t = m["t"]
        if np.ndim(t) == 0:  # limiar global: linha
            ax.axvline(t, color=COLORS[key], lw=2, label=f"{key} = {t:.3f}")
        else:                # limiar local: mediana + faixa P5–P95
            p5, med, p95 = np.percentile(t, (5, 50, 95))
            ax.axvspan(p5, p95, color=COLORS[key], alpha=0.12)
            ax.axvline(med, color=COLORS[key], lw=2, ls="--",
                       label=f"{key} med. = {med:.3f} [{p5:.2f}, {p95:.2f}]")
    ax.set_yscale("log")
    ax.set_xlabel("Intensidade (0 = preto, 1 = branco)")
    ax.set_ylabel("Pixels (log)")
    if legend:
        ax.legend(fontsize=6)


n = len(results)
ncols = min(n, 3)
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)
for ax, r in zip(axes.flat, results):
    plot_histogram(ax, r)
    ax.set_title(r["name"])
for ax in axes.flat[n:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "histogramas.png", dpi=200)
plt.show()


def error_map(pred, gt):
    out = np.ones(gt.shape + (3,))
    out[pred & gt] = (0, 0, 0)
    out[pred & ~gt] = (0.85, 0.1, 0.1)
    out[~pred & gt] = (0.1, 0.3, 0.9)
    return out


M = len(METHOD_NAMES)
for r in results:
    fig = plt.figure(figsize=(3.6 * (M + 1), 7))
    gs = GridSpec(2, M + 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(r["gray"], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Original")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(~r["gt"], cmap="gray", interpolation="nearest")
    ax.set_title("Ground truth")
    ax.axis("off")

    for k, key in enumerate(METHOD_NAMES):
        m = r["methods"][key]
        met = m["metrics"]
        ax = fig.add_subplot(gs[0, k + 1])
        ax.imshow(~m["mask"], cmap="gray", interpolation="nearest")
        ax.set_title(f"{key}\nFM {met['FM']:.1f} | PSNR {met['PSNR']:.1f} | DRD {met['DRD']:.1f}",
                     fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[1, k + 1])
        ax.imshow(error_map(m["mask"], r["gt"]), interpolation="nearest")
        ax.set_title(f"Erros {key} (vermelho FP, azul FN)", fontsize=8)
        ax.axis("off")

    fig.suptitle(r["name"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"comparacao_{r['name']}.png", dpi=200)
    plt.show()


# 3) Distribuição das métricas por método
METRICS = [("FM", "F-Measure (%) ↑"), ("PSNR", "PSNR (dB) ↑"), ("DRD", "DRD ↓")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (key, label) in zip(axes, METRICS):
    data = [[r["methods"][m]["metrics"][key] for r in results] for m in METHOD_NAMES]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, M + 1), METHOD_NAMES)
    for i, vals in enumerate(data, start=1):  # pontos individuais (n pequeno)
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, s=12, color=COLORS[METHOD_NAMES[i - 1]], zorder=3)
    ax.set_title(label)
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "metricas_boxplot.png", dpi=200)
plt.show()


# 4) Tabelas
METRIC_KEYS = ["FM", "Precisao", "Revocacao", "PSNR", "DRD"]

with open(OUT_DIR / "metricas_por_imagem.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "metodo", "limiar", *METRIC_KEYS])
    for r in results:
        for key in METHOD_NAMES:
            m = r["methods"][key]
            t = m["t"] if np.ndim(m["t"]) == 0 else np.median(m["t"])
            w.writerow([r["name"], key, f"{t:.4f}",
                        *[f"{m['metrics'][k]:.3f}" for k in METRIC_KEYS]])

with open(OUT_DIR / "metricas_resumo.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metodo", *[f"{k}_{s}" for k in METRIC_KEYS for s in ("media", "dp")]])
    print(f"\n{'Método':<10}" + "".join(f"{k:>18}" for k in METRIC_KEYS))
    for key in METHOD_NAMES:
        row, txt = [key], f"{key:<10}"
        for k in METRIC_KEYS:
            vals = np.array([r["methods"][key]["metrics"][k] for r in results])
            mu, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            row += [f"{mu:.3f}", f"{sd:.3f}"]
            txt += f"{mu:>10.2f} ± {sd:<5.2f}"
        w.writerow(row)
        print(txt)

print(f"\nResultados salvos em {OUT_DIR.resolve()}")