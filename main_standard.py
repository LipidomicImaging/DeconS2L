import os
import re
import ast
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from tqdm import tqdm
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF, PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning

from config_neg import Cfg

warnings.filterwarnings("ignore")


def align_data_to_target_axis(
    source_data,
    source_axis,
    target_axis,
    ppm_tol=10.0,
    fill_value=0.0,
    verbose=True
):
    """
    将源数据 (source_data) 映射到目标轴 (target_axis) 上。
    未匹配到的源数据(如同位素)将被丢弃。

    Args:
        source_data: (H, W, C) 或 (C, H, W) 或 (C,) 的 numpy array
        source_axis: (C_source,) 对应的源 m/z 轴
        target_axis: (C_target,) 目标 m/z 轴
        ppm_tol: 匹配容差 (PPM)
        fill_value: 如果目标轴的某个点在源轴里找不到对应，填充值

    Returns:
        target_data: 形状与 source_data 对应，但 C 维变成 len(target_axis)
    """
    input_is_chw = False
    data = source_data.copy()

    if data.shape[-1] == len(source_axis):
        pass
    elif data.shape[0] == len(source_axis):
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))  # CHW -> HWC
            input_is_chw = True
        elif data.ndim == 1:
            pass
    else:
        raise ValueError(f"数据维度 {data.shape} 与源轴长度 {len(source_axis)} 不匹配")

    if data.ndim == 3:
        H, W, _ = data.shape
        target_data = np.full((H, W, len(target_axis)), fill_value, dtype=data.dtype)
    else:
        target_data = np.full((len(target_axis),), fill_value, dtype=data.dtype)

    source_axis = np.array(source_axis)
    target_axis = np.array(target_axis)

    matched_count = 0

    for i, target_mz in enumerate(target_axis):
        idx = np.searchsorted(source_axis, target_mz)

        candidates = []
        if idx < len(source_axis):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        best_source_idx = -1
        min_dist = float("inf")

        for cand in candidates:
            src_mz = source_axis[cand]
            diff = abs(src_mz - target_mz)
            ppm = (diff / target_mz) * 1e6

            if ppm <= ppm_tol and ppm < min_dist:
                min_dist = ppm
                best_source_idx = cand

        if best_source_idx != -1:
            if data.ndim == 3:
                target_data[..., i] = data[..., best_source_idx]
            else:
                target_data[i] = data[best_source_idx]
            matched_count += 1

    if input_is_chw and target_data.ndim == 3:
        target_data = np.transpose(target_data, (2, 0, 1))

    if verbose:
        print("Alignment Report:")
        print(f"  Target Axis Length: {len(target_axis)}")
        print(f"  Source Axis Length: {len(source_axis)}")
        print(f"  Matched Peaks:      {matched_count}")
        print(f"  Dropped (Isotopes): {len(source_axis) - matched_count} (Estimated)")

    return target_data


def independent_isotope_removal_debug(data_cube, mz_axis, ppm_tol=20.0, max_isotope_level=2):
    """
    带 Debug 打印的独立同位素去除函数。
    """
    print(f">>> [DEBUG模式] 启动同位素清洗 (Max level: {max_isotope_level})...")

    H, W, C = data_cube.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.from_numpy(data_cube).permute(2, 0, 1).to(device).float()

    C13_DIFF = 1.003355
    cnt_removed = 0

    print_limit = 50
    print_count = 0

    for i in range(C):
        current_mz = mz_axis[i]

        if data[i].max() < 1e-6:
            continue

        n_carbon = current_mz * 0.062
        p = n_carbon * 0.011

        ratios = [
            min(p * 1.3, 0.45),               # M+1 ratio
            min(0.45 * p**2 * 1.3, 0.15)     # M+2 ratio
        ]

        for k in range(1, max_isotope_level + 1):
            target_mz = current_mz + (k * C13_DIFF)

            idx = np.searchsorted(mz_axis, target_mz)
            best_j = -1
            min_dist = float("inf")

            candidates = []
            if idx < C:
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)

            for candidate_j in candidates:
                diff = abs(mz_axis[candidate_j] - target_mz)
                ppm_err = (diff / target_mz) * 1e6
                if ppm_err <= ppm_tol and ppm_err < min_dist:
                    min_dist = ppm_err
                    best_j = candidate_j

            if best_j == -1:
                if print_count < print_limit:
                    print(f"  [M={current_mz:.4f}] 正在找 M+{k} (目标 {target_mz:.4f})... ❌ 未找到，跳过。")
                continue

            if best_j == i:
                continue

            current_ratio = ratios[k - 1] if (k - 1) < len(ratios) else 0.0

            if data[best_j].max() > 0:
                if print_count < 10000:
                    found_mz = mz_axis[best_j]
                    print(f"  [M={current_mz:.4f}] 正在找 M+{k}... ✅ 找到 (idx={best_j}, m/z={found_mz:.4f})")
                    print(f"      -> 使用比例: ratios[{k-1}] = {current_ratio:.6f}")
                    print(f"      -> 执行: data[{best_j}] = data[{best_j}] - data[{i}] * {current_ratio:.6f}")
                    print_count += 1

            img_source = data[i]
            img_target = data[best_j]
            to_subtract = img_source * current_ratio
            data[best_j] = torch.relu(img_target - to_subtract)
            cnt_removed += 1

    print(f">>> 清洗完成。共执行 {cnt_removed} 次减法。")
    if print_count >= 100000:
        print(f"    (注：为了防止控制台卡死，只打印了前 {print_limit} 条详细日志)")

    return data.permute(1, 2, 0).cpu().numpy()


def plot_deconvolution_results(model, img_shape, mz_axis, component_names=None):
    """
    展示每个解卷积组分的空间分布和对应纯净谱图。
    """
    C = model.C_
    S = model.S_
    n_comps = C.shape[1]

    if component_names is None:
        component_names = [f"Component {i+1}" for i in range(n_comps)]

    fig = plt.figure(figsize=(15, 3 * n_comps))
    gs = gridspec.GridSpec(n_comps, 2, width_ratios=[1, 2])

    for i in range(n_comps):
        ax_map = plt.subplot(gs[i, 0])
        spatial_img = C[:, i].reshape(img_shape)

        vmax = np.percentile(spatial_img, 99.5)
        im = ax_map.imshow(spatial_img, cmap="viridis", vmin=0, vmax=vmax)
        ax_map.set_title(f"{component_names[i]}\nSpatial Distribution", fontsize=10, fontweight="bold")
        ax_map.axis("off")
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

        ax_spec = plt.subplot(gs[i, 1])
        spectrum = S[i, :]

        markerline, stemlines, baseline = ax_spec.stem(
            mz_axis, spectrum, linefmt="b-", markerfmt="bo", basefmt=" "
        )
        plt.setp(stemlines, "linewidth", 1)
        plt.setp(markerline, "markersize", 3)

        ax_spec.set_title("Deconvolved MS/MS Spectrum", fontsize=10)
        ax_spec.set_ylabel("Intensity")
        if i == n_comps - 1:
            ax_spec.set_xlabel("m/z")

        top_indices = np.argsort(spectrum)[-15:]
        for idx in top_indices:
            if spectrum[idx] > 0.1 * np.max(spectrum):
                ax_spec.text(
                    mz_axis[idx],
                    spectrum[idx],
                    f"{mz_axis[idx]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

        ax_spec.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================================================
# 初始参数与数据加载
# =========================================================
zL = np.polyfit([327.2329, 480.3095, 790.53923], [0.009, 0.013, 0.02], 1)
zR = np.polyfit([327.2329, 480.3095, 790.53923], [0.012, 0.020, 0.03], 1)
ppL = np.poly1d(zL)
ppR = np.poly1d(zR)

H = 16
W = 10

fwhm_func_left = lambda mz: ppL(mz)
fwhm_func_right = lambda mz: ppR(mz)

META = np.load(Cfg.meta_path, allow_pickle=True)
mz_slim = META[0]["m_z_values"]

spec = np.load(Cfg.raw_spectrum_path)
spec = np.where(spec > 10, 1, 0)

my_data_cube = np.load(Cfg.raw_data_path)
fragment_mzs = np.load(Cfg.mz_path)

print(my_data_cube.shape, fragment_mzs.shape)

bac = np.where(np.sum(my_data_cube[:H * W, :], axis=1) < 800)[0]
fore = np.where(np.sum(my_data_cube[:H * W, :], axis=1) >= 800)[0]

my_data_cube[bac, :] = 0
my_data_cube = my_data_cube[:H * W, :].reshape(H, W, -1)

print(my_data_cube.shape, fragment_mzs.shape)

cleaned_data = independent_isotope_removal_debug(my_data_cube, fragment_mzs, ppm_tol=100)

summed_data = np.sum(cleaned_data, axis=0)

cond1 = np.average(summed_data, axis=0) > 10
cond2 = np.max(cleaned_data.reshape(H * W, -1), axis=0) > 10000000000000000

kept = np.unique(np.where(cond1 | cond2)[0])
cleaned_data = cleaned_data[:, :, kept]

print(cleaned_data.shape)

fragment_mzs = fragment_mzs[kept]
print(cleaned_data.shape, fragment_mzs.shape)

data_log = (torch.Tensor(cleaned_data)) / np.sum(cleaned_data, axis=2, keepdims=True) * 1000
print(data_log.shape, fragment_mzs.shape)

data_ = align_data_to_target_axis(data_log.detach().cpu().numpy(), fragment_mzs, mz_slim)

spectrum = []
ISO_LIST = []
adduct_maps_list = []

data_ = data_.reshape(H * W, -1)
kept_ = np.where(np.average(data_, axis=0) > 2)[0]

mz_slim = mz_slim[kept_]
data_ = data_[:, kept_]

comp_names = []
pre = []

for iso in META:
    spectrum.append(iso["intensities"])
    comp_names.append(iso["lipid_composition"])
    pre.append(iso["precursor_mz"])

spectrum_full = np.array(spectrum)
spec = spectrum_full
spec = np.where(spectrum_full > 1, 1, 0)

min_peaks = 3
precursor_idx = np.where(mz_slim > 770)[0]

alpha = 0.001
smooth_strength = 3

real_mask = spec[:, kept_].T
max_val = np.max(data_)
scale_factor = 1000.0 / max_val
data_scaled = data_ * scale_factor


# =========================================================
# 0. 辅助工具函数
# =========================================================
def smart_select_pixels(data_matrix, n_select=5000, n_components=50):
    """
    基于杠杆分数 (Leverage Score) 采样。
    """
    if data_matrix.shape[0] <= n_select:
        return np.arange(data_matrix.shape[0])

    print(f"   [Smart Sampling] SVD sampling on {data_matrix.shape}...")
    U, S, Vt = randomized_svd(data_matrix, n_components=n_components, random_state=42)
    leverage_scores = np.sum(U**2, axis=1)
    probs = leverage_scores / np.sum(leverage_scores)

    selected_indices = np.random.choice(
        data_matrix.shape[0],
        size=n_select,
        replace=False,
        p=probs
    )
    return selected_indices


def get_adaptive_cutoff(contributions):
    """
    自适应寻找 Major/Minor 的分割点。
    """
    n = len(contributions)
    if n < 2:
        return 0

    cumsum = np.cumsum(contributions)
    cumsum_norm = cumsum / cumsum[-1]

    x = np.linspace(0, 1, n)
    distances = cumsum_norm - x
    elbow_idx = np.argmax(distances)

    return elbow_idx


def visualize_clusters_2d(labels, H, W, fore_mask, n_clusters):
    """
    将聚类标签映射回 2D 空间并绘图。
    """
    cluster_map_1d = np.full(H * W, np.nan)

    if fore_mask.ndim == 2:
        fore_flat = fore_mask.flatten()
        cluster_map_1d[fore_flat] = labels
    else:
        cluster_map_1d[fore_mask] = labels

    cluster_map_2d = cluster_map_1d.reshape(H, W)

    plt.figure(figsize=(10, 8), dpi=120)

    if n_clusters <= 10:
        cmap = plt.get_cmap("tab10", n_clusters)
    else:
        cmap = plt.get_cmap("viridis", n_clusters)

    cmap.set_bad(color="black")

    masked_data = np.ma.masked_invalid(cluster_map_2d)
    im = plt.imshow(masked_data, cmap=cmap, interpolation="nearest")

    cbar = plt.colorbar(im, ticks=np.arange(n_clusters), shrink=0.8)
    cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    cbar.set_label("Spatial Segments")

    plt.title(f"Spatial Clustering Visualization (K={n_clusters})", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return cluster_map_2d


# =========================================================
# 1. 核心解混器
# =========================================================
class SegmentedUnmixer:
    def __init__(self, mz_axis, precursor_mzs, mz_tol=0.01, iso_exclusion=20.0):
        self.mz_axis = mz_axis
        self.precursor_mzs = np.array(precursor_mzs)
        self.mz_tol = mz_tol
        self.iso_exclusion = iso_exclusion

        self.precursor_col_idxs = np.array([self._get_col_idx(p) for p in self.precursor_mzs])
        self.valid_precursor_mask = (self.precursor_col_idxs != -1)

    def _get_col_idx(self, target_mz):
        diff = np.abs(self.mz_axis - target_mz)
        min_idx = np.argmin(diff)
        return min_idx if diff[min_idx] <= self.mz_tol else -1

    def _apply_top_k_truncation(self, coefs, k):
        if np.count_nonzero(coefs) <= k:
            return coefs
        threshold_idx = np.argpartition(coefs, -k)[-k]
        threshold_val = coefs[threshold_idx]
        coefs[coefs < threshold_val] = 0
        return coefs

    def _core_solve_routine(self, X_input, y_input, alpha, l1_ratio, top_k):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            positive=True,
            max_iter=50000,
            copy_X=False,
            tol=1e-9
        )
        model.fit(X_input, y_input)

        coefs = model.coef_
        coefs = self._apply_top_k_truncation(coefs, top_k)

        support = np.where(coefs > 0)[0]
        if len(support) > 0:
            X_supp = X_input[:, support]
            lr = LinearRegression(fit_intercept=False, positive=True)
            lr.fit(X_supp, y_input)
            coefs[support] = lr.coef_

        return coefs

    def solve(self, Y_sub):
        n_pixels, n_bins = Y_sub.shape
        n_pre = len(self.precursor_mzs)
        W_out = np.zeros((n_pre, n_bins), dtype=np.float32)

        current_pre_idxs = np.where(self.valid_precursor_mask)[0]
        cols_to_fetch = self.precursor_col_idxs[current_pre_idxs]
        X_raw = Y_sub[:, cols_to_fetch]

        col_sums = np.sum(X_raw, axis=0)
        valid_cols = col_sums > 1e-9
        if np.sum(valid_cols) == 0:
            return W_out

        X_active = X_raw[:, valid_cols]
        final_pre_idxs = current_pre_idxs[valid_cols]
        forbidden_frag_indices = set(self.precursor_col_idxs[final_pre_idxs])

        target_indices = np.where(np.sum(Y_sub, axis=0) > 1e-9)[0]

        for frag_idx in target_indices:
            if frag_idx in forbidden_frag_indices:
                continue

            frag_mz = self.mz_axis[frag_idx]
            y_frag = Y_sub[:, frag_idx]

            alpha, l1, top_k = 1e-4, 0.9, 2

            min_pre_mz = frag_mz + self.iso_exclusion
            valid_mask = self.precursor_mzs[final_pre_idxs] > min_pre_mz
            if np.sum(valid_mask) == 0:
                continue

            X_curr = X_active[:, valid_mask]
            x_max = np.max(X_curr, axis=0) + 1e-9
            y_max = np.max(y_frag) + 1e-9

            coefs = self._core_solve_routine(X_curr / x_max, y_frag / y_max, alpha, l1, top_k)
            w_real = coefs * (y_max / x_max)

            W_out[final_pre_idxs[valid_mask], frag_idx] = w_real

        return W_out

    def solve_peeling(self, Y_sub, major_indices, minor_indices):
        n_pixels, n_bins = Y_sub.shape
        n_pre = len(self.precursor_mzs)
        W_out = np.zeros((n_pre, n_bins), dtype=np.float32)

        def get_X(indices):
            cols = self.precursor_col_idxs[indices]
            valid = cols != -1
            if np.sum(valid) == 0:
                return None, None
            return Y_sub[:, cols[valid]], indices[valid]

        X_major, idx_major_global = get_X(major_indices)
        X_minor, idx_minor_global = get_X(minor_indices)

        target_indices = np.where(np.sum(Y_sub, axis=0) > 1e-9)[0]
        forbidden_frag_indices = set(self.precursor_col_idxs[self.valid_precursor_mask])

        for frag_idx in target_indices:
            if frag_idx in forbidden_frag_indices:
                continue

            frag_mz = self.mz_axis[frag_idx]
            y_frag = Y_sub[:, frag_idx]
            y_max = np.max(y_frag) + 1e-9
            y_input = y_frag / y_max

            if frag_mz <= 500:
                alpha, l1, k = 1e-5, 0.9, 2
            elif frag_mz <= 650:
                alpha, l1, k = 1e-5, 0.9, 2
            else:
                alpha, l1, k = 1e-5, 0.9, 2

            min_pre_mz = frag_mz + self.iso_exclusion

            w_major_scaled = None
            if X_major is not None:
                mask_mz = self.precursor_mzs[idx_major_global] > min_pre_mz
                if np.sum(mask_mz) > 0:
                    X_curr = X_major[:, mask_mz]
                    curr_idxs = idx_major_global[mask_mz]
                    x_max = np.max(X_curr, axis=0) + 1e-9

                    coefs = self._core_solve_routine(X_curr / x_max, y_input, alpha, l1, k)
                    w_major_scaled = coefs
                    W_out[curr_idxs, frag_idx] = coefs * (y_max / x_max)

            y_resid_input = y_input.copy()
            if w_major_scaled is not None and np.sum(w_major_scaled) > 0:
                y_pred = (X_curr / x_max) @ w_major_scaled
                y_resid_input = np.maximum(y_input - y_pred, 0)

            if np.max(y_resid_input) < 1e-9:
                continue

            if X_minor is not None:
                mask_mz = self.precursor_mzs[idx_minor_global] > min_pre_mz
                if np.sum(mask_mz) > 0:
                    X_curr = X_minor[:, mask_mz]
                    curr_idxs = idx_minor_global[mask_mz]
                    x_max = np.max(X_curr, axis=0) + 1e-9

                    coefs = self._core_solve_routine(X_curr / x_max, y_resid_input, alpha * 0.1, l1, k)
                    W_out[curr_idxs, frag_idx] = coefs * (y_max / x_max)

        return W_out


# =========================================================
# 3. 主流程
# =========================================================
pre = [770.570894, 758.489]

data_fore = data_scaled[fore, :]
row_sums = np.sum(data_fore, axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0

solver = SegmentedUnmixer(mz_axis=mz_slim, precursor_mzs=np.unique(np.around(pre, 3)))

print("\n=== Step 1: 全局求解 & 划分 Major/Minor ===")
global_idx = smart_select_pixels(data_fore, n_select=3000)
W_global = solver.solve(data_fore[global_idx])

precursor_contribution = np.sum(W_global, axis=1)
sorted_idx = np.argsort(precursor_contribution)[::-1]
sorted_scores = precursor_contribution[sorted_idx]

cutoff_idx = get_adaptive_cutoff(sorted_scores)

major_indices = sorted_idx[:cutoff_idx + 1]
minor_indices = sorted_idx[cutoff_idx + 1:]
minor_indices = minor_indices[precursor_contribution[minor_indices] > 1e-9]

print(f"   [Adaptive] Major Precursors: {len(major_indices)}")
print(f"   [Adaptive] Minor Precursors: {len(minor_indices)}")

print("\n=== Step 2: 聚类 (含可视化) ===")
n_clusters = 4
top_feat = np.argsort(np.sum(data_fore, axis=0))[-300:]
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(data_fore[:, top_feat])
labels = kmeans.labels_

try:
    print("   Visualizing clusters...")
    cluster_map = visualize_clusters_2d(labels, H, W, fore, n_clusters)
except NameError as e:
    print(f"   [Error] Visualization failed: {e}")
except Exception as e:
    print(f"   [Error] Visualization error: {e}")

print("\n=== [Fixed] Step 3: 联合求解 (Joint Solving) ===")
results_dict = {}

pre_array = np.sort(np.unique(np.array(pre)))
solver = SegmentedUnmixer(mz_axis=mz_slim, precursor_mzs=pre_array)

for k in range(n_clusters):
    mask = (labels == k)
    n_pixels = np.sum(mask)

    if n_pixels < 20:
        continue

    print(f"  -> Cluster {k}: Joint Analysis ({n_pixels} pixels)...")

    Y_sub = data_fore[mask, :]

    if Y_sub.shape[0] > 200:
        idx = smart_select_pixels(Y_sub, n_select=50)
        Y_sub = Y_sub[idx, :]

    W_k = solver.solve(Y_sub)
    results_dict[k] = W_k

print("\nDone. Results stored in 'results_dict'.")
print("现在去画图看看，PC和PS的独家碎片应该各回各家了。")


# =========================================================
# 结果导出与后处理
# =========================================================
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT_SAVE_DIR = "Strict_FragOnly_Solver_Results"
if not os.path.exists(ROOT_SAVE_DIR):
    os.makedirs(ROOT_SAVE_DIR)


def convert_custom_data_to_db(data_list):
    """将 META 列表转换为 DataFrame"""
    db_rows = []
    print(f"   [DB] Converting {len(data_list)} META entries...")

    for item in data_list:
        name = item.get("lipid_composition", "Unknown")
        adduct = item.get("adduct", "")
        pre_mz = float(item.get("precursor_mz", 0.0))
        mz_axis = np.array(item["m_z_values"])
        intensities = np.array(item["intensities"])

        valid_indices = np.where(intensities > 1e-9)[0]
        compact_spectrum = []

        if len(valid_indices) > 0:
            for m, i in zip(mz_axis[valid_indices], intensities[valid_indices]):
                compact_spectrum.append([float(m), float(i)])

        db_rows.append({
            "Lipid_Name": name,
            "Precursor_mz": pre_mz,
            "Adduct": adduct,
            "msms_m0_noiso": compact_spectrum
        })

    return pd.DataFrame(db_rows)


def find_lipid_candidates(
    precursor_mz,
    mz_axis,
    spectrum,
    database,
    offset_dict=None,
    ppm_tol=20,
    top_k=1
):
    """支持 Offset (源内裂解) 的鉴定逻辑"""
    if offset_dict is None:
        offset_dict = {}

    search_mz = precursor_mz
    offset_used = 0.0

    for k, v in offset_dict.items():
        if abs(precursor_mz - k) < 0.2:
            offset_used = v
            search_mz = precursor_mz + v
            break

    candidates_df = database[np.abs(database["Precursor_mz"] - search_mz) < 0.05].copy()
    if candidates_df.empty:
        return [], offset_used

    max_int = np.max(spectrum)
    if max_int < 1e-9:
        return [], offset_used

    valid_indices = np.where(spectrum > 0.01 * max_int)[0]
    valid_mzs = mz_axis[valid_indices]

    matched_candidates = []
    for _, row in candidates_df.iterrows():
        std_frags = row["msms_m0_noiso"]
        if not std_frags:
            continue

        current_indices = []
        for frag in std_frags:
            std_mz = float(frag[0])
            diffs = np.abs(valid_mzs - std_mz)
            if len(diffs) == 0:
                continue

            min_idx_local = np.argmin(diffs)
            if (diffs[min_idx_local] / std_mz) * 1e6 <= ppm_tol:
                current_indices.append(valid_indices[min_idx_local])

        current_indices = list(set(current_indices))
        if len(current_indices) >= 2:
            score = len(current_indices) + (np.sum(spectrum[current_indices]) / (np.sum(spectrum) + 1e-9)) * 5
            matched_candidates.append({
                "name": row["Lipid_Name"],
                "indices": current_indices,
                "score": score,
                "pmz": row["Precursor_mz"]
            })

    matched_candidates.sort(key=lambda x: x["score"], reverse=True)
    return matched_candidates[:top_k], offset_used


class LipidDataHandler:
    def __init__(self, data_matrix, mz_axis, H, W, fore_indices):
        self.H = int(H)
        self.W = int(W)
        self.mz_axis = np.array(mz_axis)
        self.n_total = self.H * self.W
        self.valid_indices = np.array(fore_indices).flatten().astype(int)
        self.n_fore = len(self.valid_indices)

        if data_matrix.ndim == 3:
            data_flat = data_matrix.reshape(-1, data_matrix.shape[-1])
            self.data = data_flat[self.valid_indices, :]
        else:
            self.data = data_matrix[self.valid_indices, :] if data_matrix.shape[0] == self.n_total else data_matrix

        print(f"✅ [Handler] Ready. Shape: {self.data.shape}")

    def get_data_matrix_integrated(self, target_mzs, tol=0.02):
        n_pixels = self.data.shape[0]
        n_targets = len(target_mzs)
        V_out = np.zeros((n_pixels, n_targets), dtype=np.float32)
        valid_mz_list = []

        for i, tmz in enumerate(target_mzs):
            mask = np.abs(self.mz_axis - tmz) <= tol
            bin_indices = np.where(mask)[0]
            if len(bin_indices) > 0:
                V_out[:, i] = np.sum(self.data[:, bin_indices], axis=1)
            valid_mz_list.append(tmz)

        return V_out, valid_mz_list


class FixedDictionaryProjector:
    def solve(self, V, H_fixed, n_iter=500):
        n_pixels = V.shape[0]
        n_components = H_fixed.shape[0]

        W = np.random.rand(n_pixels, n_components).astype(np.float32) + 0.1
        HHt = H_fixed @ H_fixed.T
        eps = 1e-9

        for i in tqdm(range(n_iter), desc="Solving Spatial Map"):
            numer = V @ H_fixed.T
            denom = W @ HHt + eps
            W *= (numer / denom)

        return W


custom_db = convert_custom_data_to_db(META)
handler = LipidDataHandler(data_scaled, mz_slim, H, W, fore)
offset_config = {770.570894: 50.0, 671.466: 87.03}

if "pre" in globals():
    target_precursor_mzs = np.unique(np.around(pre, 3))
else:
    target_precursor_mzs = np.unique(np.around(list(results_dict.keys())[0]["pmz"], 3))

print("\n=== Step 1: Identification with Offset Strategy ===")
best_lipid_data = {}

for k, W_matrix in tqdm(results_dict.items(), desc="Analyzing Clusters"):
    active_indices = np.where(np.sum(W_matrix, axis=1) > 1e-9)[0]

    for idx in active_indices:
        try:
            pmz_obs = target_precursor_mzs[idx]
        except:
            continue

        spectrum = W_matrix[idx]
        current_total_int = np.sum(spectrum)

        candidates, offset_used = find_lipid_candidates(
            pmz_obs, mz_slim, spectrum, custom_db, offset_dict=offset_config
        )

        for cand in candidates:
            name = cand["name"]
            if offset_used > 0:
                name += f" [ISD+{offset_used:.0f}]"

            matched_indices = cand["indices"]
            temp_frag_dict = {round(float(mz_slim[i]), 4): spectrum[i] for i in matched_indices}

            if temp_frag_dict:
                max_f = max(temp_frag_dict.values())
                norm_frags = {m: v / max_f for m, v in temp_frag_dict.items()}
                norm_frags[round(float(pmz_obs), 4)] = 0.4

                if name not in best_lipid_data or current_total_int > best_lipid_data[name]["max_int"]:
                    best_lipid_data[name] = {
                        "max_int": current_total_int,
                        "spectrum": norm_frags,
                        "pmz": pmz_obs
                    }

lipid_id_list = sorted(list(best_lipid_data.keys()))
n_lipids = len(lipid_id_list)
print(f"   -> Found {n_lipids} Lipids")

print("\n=== Step 2: Matrix Construction ===")
all_frags = set()
for uid in lipid_id_list:
    all_frags.update(best_lipid_data[uid]["spectrum"].keys())

global_frag_axis = np.sort(list(all_frags))

V_frag_only, final_mz_channels = handler.get_data_matrix_integrated(global_frag_axis, tol=0.02)
final_mz_arr = np.array(final_mz_channels)

H_fixed = np.zeros((n_lipids, len(global_frag_axis)), dtype=np.float32)
for i, uid in enumerate(lipid_id_list):
    for mz_k, val in best_lipid_data[uid]["spectrum"].items():
        b_idx = np.argmin(np.abs(final_mz_arr - mz_k))
        if np.abs(final_mz_arr[b_idx] - mz_k) < 0.05:
            H_fixed[i, b_idx] = val

W_solved = FixedDictionaryProjector().solve(V_frag_only, H_fixed, n_iter=1000)


def save_robust_image(data_2d, save_path, cmap_name="hot"):
    """鲁棒保存函数：处理 NaN, 99.5% 分位数归一化"""
    H_img, W_img = data_2d.shape
    rgba_img = np.zeros((H_img, W_img, 4), dtype=np.float32)
    rgba_img[..., 3] = 1.0

    valid_mask = ~np.isnan(data_2d)
    if not np.any(valid_mask):
        plt.imsave(save_path, rgba_img)
        return

    valid_data = data_2d[valid_mask]
    vmax = np.percentile(valid_data, 99.5) if np.max(valid_data) > 0 else 1
    vmin = np.min(valid_data)

    if vmax <= vmin:
        vmax = vmin + 1e-9

    norm_data = np.clip((valid_data - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap(cmap_name)
    rgba_img[valid_mask] = cmap(norm_data)
    rgba_img[~valid_mask] = [0, 0, 0, 1]

    plt.imsave(save_path, rgba_img)


print("\n=== Step 4: Robust Exporting ===")
for i, uid in enumerate(tqdm(lipid_id_list, desc="Exporting")):
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", uid).replace(" ", "_")
    l_dir = os.path.join(ROOT_SAVE_DIR, f"ID{i:03d}_{safe_name}")
    os.makedirs(l_dir, exist_ok=True)

    img_flat = np.full(handler.n_total, np.nan, dtype=np.float32)
    img_flat[handler.valid_indices] = W_solved[:, i]
    save_robust_image(img_flat.reshape(handler.H, handler.W), os.path.join(l_dir, "Spatial_Deconv.png"))

    target_pmz = best_lipid_data[uid]["pmz"]
    raw_v, _ = handler.get_data_matrix_integrated([target_pmz])
    img_flat_pre = np.full(handler.n_total, np.nan, dtype=np.float32)
    img_flat_pre[handler.valid_indices] = raw_v[:, 0]
    save_robust_image(
        img_flat_pre.reshape(handler.H, handler.W),
        os.path.join(l_dir, "Precursor_Raw.png"),
        cmap_name="gray"
    )

    plt.figure(figsize=(6, 3))
    indices = np.where(H_fixed[i, :] > 0)[0]
    if len(indices) > 0:
        mzs, ints = final_mz_arr[indices], H_fixed[i, indices]
        plt.stem(mzs, ints, linefmt="r-", markerfmt="ro", basefmt=" ")
        for m, v in zip(mzs, ints):
            plt.text(m, v, f"{m:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.title(f"Ref Spectrum: {uid}", fontsize=8)
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(l_dir, "Ref_Spectrum.png"), bbox_inches="tight", dpi=150)
    plt.close()

print(f"✅ 完成！结果已保存至: {ROOT_SAVE_DIR}")
