import numpy as np
from scipy.special import digamma, polygamma #gamma_gamma 那個會用到
from scipy.stats import gamma as sp_gamma
from sklearn.metrics import f1_score


# === 算 F1 ===
def f1_for_threshold(thr, X, y):
    if thr is None:
        return None
    y_pred = (X < thr).astype(int)  # < thr 視為調整過 (positive class)
    return f1_score(y, y_pred)

# === 算 F1(0是正類) ===
def f1_for_threshold_0(thr, X, y):
    if thr is None:
        return None
    y_pred = (X < thr).astype(int)  # < thr 視為調整過 (positive class)
    return f1_score(y, y_pred, pos_label=0)

# === 算 F1(都考慮) ===
def f1_for_threshold_wei(thr, X, y):
    if thr is None:
        return None
    y_pred = (X < thr).astype(int)  # < thr 視為調整過 (positive class)
    return f1_score(y, y_pred, average='weighted')

def f1_for_threshold_mix(thr1, thr2, vals1, vals2, y_true):
    """
    thr1: 第一個異常的門檻 (小於 => 視為異常)
    thr2: 第二個異常的門檻 (小於 => 視為異常)
    vals1: 第一個異常的分數 (例如 light_score)
    vals2: 第二個異常的分數 (例如 blur_score)
    y_true: 真實標籤 (1=異常, 0=正常)
    """
    if thr1 is None or thr2 is None:
        return None, None

    cond1 = (vals1 < thr1)  # 第一個異常
    cond2 = (vals2 < thr2)  # 第二個異常

    # 只要符合任一條件，就判斷為異常
    y_pred = (cond1 | cond2).astype(int)

    return f1_score(y_true, y_pred), y_pred


def f1_for_threshold_mix_0(thr1, thr2, vals1, vals2, y_true):
    if thr1 is None or thr2 is None:
        return None, None
    cond1 = (vals1 < thr1)  # 第一個異常
    cond2 = (vals2 < thr2)  # 第二個異常

    # 只要符合任一條件，就判斷為異常
    y_pred = (cond1 | cond2).astype(int)
    return f1_score(y_true, y_pred, pos_label=0), y_pred

def f1_for_threshold_mix_wei(thr1, thr2, vals1, vals2, y_true):
    if thr1 is None or thr2 is None:
        return None, None
    cond1 = (vals1 < thr1)  # 第一個異常
    cond2 = (vals2 < thr2)  # 第二個異常

    # 只要符合任一條件，就判斷為異常
    y_pred = (cond1 | cond2).astype(int)
    return f1_score(y_true, y_pred, average='weighted'), y_pred


# --------------- (A) 用 scikit-image 的 Otsu（若可用） ---------------
def otsu_skimage(values) -> float | None:
  '''
  1. 讀取一維數據（例如亮度分數、哈明距離），並自動建立直方圖。
  2. 遍歷所有可能的分割點（候選閾值）。
  3. 對每個候選閾值，把數據切成兩組：小於等於閾值的一組（類別 0）、大於閾值的一組（類別 1）。
  4. 計算這兩組的「類內方差」以及「類間方差」。
  5. 找到能 最大化類間方差（等價於最小化類內方差）的那個閾值。
  '''
  try:
      from skimage.filters import threshold_otsu
      # Otsu 可以直接吃連續值，不必先量化
      return float(threshold_otsu(values))
  except Exception as e:
      return None

# --------------- (B) 純 NumPy 的 Otsu（無相依、可移植） ---------------
def otsu_numpy(values, nbins=256) -> float | None:
    """
    對連續數據計算 Otsu 門檻。
    values: 1D array，自動以資料極值建直方圖
    nbins: 直方圖箱數
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)] #把 NaN 去掉
    if v.size == 0:
        return None

    vmin, vmax = v.min(), v.max() #抓資料最小/最大值，準備設定直方圖的範圍
    if vmax <= vmin: #所有值都相同就回傳那個數當門檻。
        return float(vmin)

    # 建直方圖（密度不用，採計數）
    counts, bin_edges = np.histogram(v, bins=nbins, range=(vmin, vmax))
    counts = counts.astype(np.float64) #把計數轉成浮點，方便後續「機率」與加權平均運算（避免整數除法問題）。

    # 機率與累積統計
    p = counts / counts.sum() #每個箱的比例
    omega = np.cumsum(p)   #omega[t] 是從最小值到第 t 個箱的累積機率
    mu_bins = (bin_edges[:-1] + bin_edges[1:]) * 0.5  #算每個箱子的 中心值（用左右邊界平均）。這代表該箱的代表數值
    mu = np.cumsum(p * mu_bins)  # 累積均值 (p * mu_bins 是每個箱對「整體平均」的加權貢獻；np.cumsum(...) 讓你得到 到第 t 箱為止的累積平均分子)
    mu_t = mu_bins.dot(p)  # 全域均值 (整體期望值)

    # 類間方差：sigma_b^2 = (mu_t * omega - mu)^2 / (omega * (1 - omega))
    denom = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'): #np.errstate 關閉 0 除錯誤警告，並把分母 ≤ 0 的位置（類別一邊沒有樣本）直接設為 0，表示那個 t 不是有效分割
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        sigma_b2[denom <= 0] = 0.0

    # 找到最大類間方差的 bin 當作門檻
    idx = np.argmax(sigma_b2)
    thr = mu_bins[idx]
    return float(thr)

# ====== Kittler–Illingworth Minimum Error Threshold (純 NumPy) ======
def ki_min_error_threshold(values, bins=256, data_range=None, eps=1e-12) -> float | None:
    """
    無監督最小錯誤率閥值法（Kittler & Illingworth, 1986）
    values: 1D array (連續值)
    bins:   可給 int 或與 plt.hist 相同的 bin 邊界陣列
    data_range: (min, max)；若 bins 為邊界陣列就不用
    回傳: 門檻值 (float) 或 None（無法計算時）
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)] #去掉 NaN。
    if v.size == 0:
        return None

    # 直方圖（用與繪圖一致的 bins 可視覺對齊）
    if isinstance(bins, int):
        counts, edges = np.histogram(v, bins=bins, range=data_range)
    else:
        counts, edges = np.histogram(v, bins=bins)
    counts = counts.astype(np.float64)
    n = counts.sum()
    if n == 0:
        return None

    p = counts / n  #每個 bin 的相對頻率（機率質量函數）
    centers = (edges[:-1] + edges[1:]) * 0.5

    # 累積量
    P1 = np.cumsum(p)            # 左側權重 ω0(t) (把閾值切在 bin t 時，左邊的機率質量（類別 0 的權重）)
    P2 = 1.0 - P1             # 右側權重 ω1(t) (右邊的權重 = 1 - P1)
    m1 = np.cumsum(p * centers)      # 左側一階動差 (左邊到 t 的累積一階動差（和）)
    mT = (p * centers).sum()        # 全域一階動差 (整體平均的分子（全域一階動差）)
    m2 = np.cumsum(p * centers**2)     # 左側二階動差 (左邊到 t 的累積二階動差)
    M2 = (p * centers**2).sum()      # 全域二階動差 (全域二階動差)

    # 條件平均
    mu1 = np.where(P1 > 0, m1 / np.maximum(P1, eps), 0.0) #把閾值設在 t 時，左側類別的平均值
    mu2 = np.where(P2 > 0, (mT - m1) / np.maximum(P2, eps), 0.0) #右側類別的平均值 (np.maximum(..., eps) 避免除以 0)

    # 條件變異數
    var1 = np.where(P1 > 0, m2 / np.maximum(P1, eps) - mu1**2, 0.0) #左側類別的條件方差
    var2 = np.where(P2 > 0, (M2 - m2) / np.maximum(P2, eps) - mu2**2, 0.0) #右側類別的條件方差

    #print(var1,var2)

    # 避免非正或零方差/權重造成 log 出錯
    valid = (P1 > 0) & (P2 > 0) & (var1 > 0) & (var2 > 0)

    if not np.any(valid):
        return None

    # 取出有效候選點的標準差（σ1, σ2）和權重（ω1, ω2）
    sigma1 = np.sqrt(var1[valid])
    sigma2 = np.sqrt(var2[valid])
    w1 = P1[valid]
    w2 = P2[valid]

    # Kittler–Illingworth 準則
    # J(t) = 1 + 2*(w1*log(sigma1) + w2*log(sigma2)) - 2*(w1*log(w1) + w2*log(w2))
    J = 1.0 + 2.0 * (w1 * np.log(np.maximum(sigma1, eps)) + w2 * np.log(np.maximum(sigma2, eps))) \
            - 2.0 * (w1 * np.log(np.maximum(w1, eps)) + w2 * np.log(np.maximum(w2, eps)))

    # 把索引對回完整 bins 座標
    valid_idx = np.flatnonzero(valid)
    best_idx = valid_idx[np.argmin(J)]

    # 用該 bin 的中心值當門檻（你也可以改用右邊界 edges[best_idx+1]）
    return float(centers[best_idx])


# --------------- (A) 用 scikit-image 的 最小交叉熵（若可用） ---------------
def li_min_cross_entropy(values, tolerance=None, initial_guess=None, iter_callback=None) -> float | None:
  '''
  1. 讀取一維數據（例如亮度分數、距離），並自動建立直方圖或直接處理原始值。
  2. 設定初始閾值（預設數據均值）。
  3. 把數據切成兩組：
    - 類別 0：小於等於閾值的數據
    - 類別 1：大於閾值的數據
  4. 計算兩組的平均值 μ0、μ1，並更新閾值：
        t_new = (μ0 - μ1) / (log(μ0) - log(μ1))
  5. 重複迭代，直到新舊閾值收斂（差異小於設定容忍度）。
  6. 收斂後的閾值即為最小交叉熵法的最佳閾值。
  '''
  try:
      from skimage.filters import threshold_li
      # Li 方法可以直接處理連續值，不必先量化
      return float(threshold_li(values, tolerance=tolerance, initial_guess=initial_guess, iter_callback=iter_callback))
  except Exception:
      return None


# ========= 最小交叉熵（Li / Li–Tam）閾值：純 NumPy 版本 =========
def li_min_cross_entropy_np(values, bins=256, data_range=None, max_iter=1000, tol=1e-7, eps=1e-12) -> float | None:
    """
    最小交叉熵（Li / Li–Tam）全域閾值（直方圖版本）。
    values: 1D array（連續值）
    bins:   int 或 一組邊界陣列（與 plt.hist 相同），建議用和繪圖相同的 bins 以便對齊
    data_range: (min, max)；若 bins 是邊界陣列就不用
    回傳: 浮點數門檻（若失敗回傳 None）
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None

    # 直方圖（與繪圖同 bins 則線會對齊）
    if isinstance(bins, int):
        counts, edges = np.histogram(v, bins=bins, range=data_range)
    else:
        counts, edges = np.histogram(v, bins=bins)
    counts = counts.astype(np.float64)
    n = counts.sum()
    if n == 0:
        return None

    p = counts / n                   # 每個 bin 的機率
    centers = (edges[:-1] + edges[1:]) * 0.5     # bin 中心
    cmin, cmax = centers.min(), centers.max()

    # 初始門檻（用全域平均或中心值中位）
    mT = np.sum(p * centers)  # 全域一階動差 (整個分布的加權平均)
    t = float(mT) if np.isfinite(mT) else float((cmin + cmax) * 0.5)  #先用這個值作為 初始門檻（如果失敗，就退而用中點）

    # Li–Tam 迭代： t_{new} = (m1 - m2) / (ln(m1) - ln(m2))
    # m1/m2 為左右兩側條件平均。避免 0、避免空側。
    for _ in range(max_iter):
        left  = centers <= t
        right = ~left #用當前的門檻 t 把直方圖分成左側（小於等於門檻）和右側（大於門檻）。
        w1, w2 = p[left].sum(), p[right].sum() #算出左邊 (w1) 和右邊 (w2) 的機率質量
        if w1 <= 0 or w2 <= 0:
            # 若切點導致某側無樣本，回退為最近非空 bin 的中心
            # 找到最接近 t 的 bin index，向左右擴展到非空側
            idx = np.searchsorted(centers, t, side='right') - 1
            idx = np.clip(idx, 0, len(centers)-1)
            # 向左找非空
            li = idx
            while li >= 0 and p[li] == 0:
                li -= 1
            # 向右找非空
            ri = idx+1
            while ri < len(centers) and p[ri] == 0:
                ri += 1
            cand = []
            if li >= 0: cand.append(centers[li])
            if ri < len(centers): cand.append(centers[ri])
            return float(cand[0]) if cand else None

        m1 = (p[left]  * centers[left]).sum()  / max(w1, eps) #計算左側和右側的條件平均值m1,m2
        m2 = (p[right] * centers[right]).sum() / max(w2, eps)

        # 避免 log(0)：若 m1/m2 太小，微量抬升
        m1_safe = max(m1, eps)
        m2_safe = max(m2, eps)
        denom = np.log(m1_safe) - np.log(m2_safe)
        if abs(denom) < eps: #denom 太接近 0 時避免數值爆炸，就退回左右均值的中點
            # 兩側均值太接近，收斂到中點
            t_new = 0.5 * (m1 + m2)
        else:
            t_new = (m1 - m2) / denom

        # 收斂判定(不斷更新 t 直到收斂)
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new

    # 夾回範圍內；若超出，選最近 bin 中心
    if not (np.isfinite(t) and cmin <= t <= cmax):
        t = float(np.clip(t, cmin, cmax))

    return float(t)


# ===================== Kapur 最大熵閾值（純 NumPy） =====================
def kapur_max_entropy(values, bins=256, data_range=None, eps=1e-12) -> float | None:
    """
    Kapur 最大熵（Maximum Entropy）全域閾值。
    values: 1D array（連續值）
    bins:   int 或 一組邊界陣列（與 plt.hist 相同），建議用和繪圖相同的 bins 以對齊
    data_range: (min, max)；若 bins 為邊界陣列就不用
    回傳: 浮點數門檻（若無法計算回傳 None）
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None

    # 直方圖（對齊你的繪圖 bins）
    if isinstance(bins, int):
        counts, edges = np.histogram(v, bins=bins, range=data_range)
    else:
        counts, edges = np.histogram(v, bins=bins)
    counts = counts.astype(np.float64)
    n = counts.sum()
    if n == 0:
        return None

    p = counts / n                 # 機率質量函數
    centers = (edges[:-1] + edges[1:]) * 0.5   # bin 中心

    # 累積機率與 p*log(p) 的累積（0*log0 視為 0）
    P1 = np.cumsum(p)                  # 左類別權重 (累積機率：P1[t]=把切點放在第 t bin 時，左側類別的權重 ω0(t))
    S  = np.where(p > 0, p * np.log(p), 0.0)   # p*log(p) (算 p_i*log(p_i)，定義 0*log(0)=0 以避免 -inf
    S_cum = np.cumsum(S)   #p*log(p) 的累積和：左側的總和 S0(t)
    S_tot = S.sum()     # 全域的 p*log(p) 總和

    # 有效候選：左右兩側權重皆 > 0
    valid = (P1 > 0) & (P1 < 1.0)
    if not np.any(valid):
        return None

    P1v = P1[valid]
    P2v = 1.0 - P1v
    S0  = S_cum[valid]   # 左側 p*log(p) 累積

    # Kapur 目標：最大化 H0 + H1
    # H0 = - sum_{i<=t} (p_i/P1) log(p_i/P1) = - S0/P1 + log(P1)
    # H1 = - sum_{i> t} (p_i/P2) log(p_i/P2) = - (S_tot - S0)/P2 + log(P2)
    H0 = - (S0 / np.maximum(P1v, eps)) + np.log(np.maximum(P1v, eps))  #左側熵
    H1 = - ((S_tot - S0) / np.maximum(P2v, eps)) + np.log(np.maximum(P2v, eps)) ## 右側熵
    J  = H0 + H1

    # 取最大熵的切點，回傳對應 bin 的中心值（也可改 edges[idx+1]）
    valid_idx = np.flatnonzero(valid)
    best_idx = valid_idx[np.argmax(J)]
    return float(centers[best_idx])


# ===================== GHT（Barron 2020）=====================
def ght_threshold(values, bins=256, data_range=None, nu=2**(29.5), tau=2**(3.125), kappa=2**(22.25), omega=2**(-3.25), eps=1e-30) -> float | None:
    """
    Generalized Histogram Thresholding.
    values: 一維數據（如 score / Hamming 距離）
    bins:   int 或 一組邊界（用與繪圖相同的 bins 就能對齊）
    data_range: (min, max)；若 bins 是邊界陣列則不需給
    超參數：
      - nu>=0, tau>=0：variance 的先驗（nu=0,kappa=0 時退化為 MET）
      - kappa>=0, omega∈[0,1]：類別權重的 Beta 先驗（kappa 大→更靠近加權分位數）（kappa 像先驗總量，omega 是先驗平均）
    回傳：單一浮點門檻；若失敗回傳 None
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None

    # 直方圖（與繪圖用同一組 bins）
    if isinstance(bins, int):
        n, edges = np.histogram(v, bins=bins, range=data_range)
    else:
        n, edges = np.histogram(v, bins=bins)
    n = n.astype(np.float64)
    if n.sum() == 0:
        return None

    # bin 的位置用中心值（GHT 允許任意 x）
    x = (edges[:-1] + edges[1:]) * 0.5

    # 累積與反向累積（與論文附錄一致）
    csum  = lambda z: np.cumsum(z)[:-1] #切在 bin t 的左側總和
    dsum  = lambda z: np.cumsum(z[::-1])[-2::-1] #切在 bin t 的右側總和
    clip  = lambda z: np.maximum(eps, z) #把數值下界夾到 eps，避免 0

    # 預先計算公共量（附錄 Algorithm 1 的 preliminaries）
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2     # 左側平方和減去均值平方 → (加權)方差乘以權重
    d1 = dsum(n * x**2) - w1 * mu1**2

    # tau 預設：用樣本標準差（數值尺度較穩），也可改用 bin 寬度
    if tau is None:
        tau = np.std(v)
        if not np.isfinite(tau) or tau <= 0:
            tau = (edges[-1] - edges[0]) / max(1, (len(edges)-1))

    # GHT 核心打分（附錄 Algorithm 1）
    v0 = clip((p0 * nu * (tau**2) + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * (tau**2) + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2.0 * (w0 + kappa * omega)      * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2.0 * (w1 + kappa * (1.0-omega)) * np.log(w1)
    score = f0 + f1

    # 與附錄一致：若有並列取平均位置；回傳對應門檻（用 x 的位置）
    valid_idx = np.arange(len(x)-1)  # 最後一個 bin 右邊沒有 split
    # 這裡 x[:-1] 與 score 對齊；取 score 最大的索引集合之 x 平均
    best_mask = (score == np.max(score))
    if not np.any(best_mask):
        return None
    return float(np.mean(x[:-1][best_mask]))


# ---------- Modified Valley-Emphasis Otsu (MVE-Otsu, 2020) ----------
def mve_otsu(values, bins=256, data_range=None, smooth_window=11, eps=1e-12) -> float | None:
    """
    Xing et al., 2020: δ(t) = σ_b^2(t) * w_v(t)
    w_v 由平滑後直方圖的「二階差分」正規化而來；以同一組 bins 與你的圖對齊。
    values: 一維連續值
    bins  : int 或 邊界陣列（建議用和繪圖相同的 bins）
    smooth_window: 平滑視窗（需奇數；論文建議 2k+1, k=5 -> 11）

    流程：把直方圖平滑 → 對平滑後的直方圖做二階差分得到“谷值權重” → 將權重乘到 Otsu 類間方差上，取目標最大的切點
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None

    # 直方圖（與繪圖用同一組 bins 對齊）
    if isinstance(bins, int):
        counts, edges = np.histogram(v, bins=bins, range=data_range)
    else:
        counts, edges = np.histogram(v, bins=bins)
    counts = counts.astype(np.float64)
    n = counts.sum()
    if n == 0:
        return None

    # 機率質量函數 p(i)
    p = counts / n

    # --- 平滑直方圖（移動平均），避免二階差分受鋸齒影響 ---
    if smooth_window is None or smooth_window < 1:
        ps = p
    else:
        if smooth_window % 2 == 0:
            smooth_window += 1  # 要奇數
        k = smooth_window // 2
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        # 以邊界延展的方式做 same-conv
        pad = np.pad(p, (k, k), mode='edge') #把 p 的左右端用「邊界值重複」的方式補 k 個元素，避免邊緣被零填充而變形
        ps = np.convolve(pad, kernel, mode='valid') #valid:核完全覆蓋到資料的區段才輸出

    # --- 二階差分（離散二階導數）：d2[i] = p[i+1] - 2 p[i] + p[i-1] ---
    d2 = np.zeros_like(ps)
    if len(ps) >= 3:
        d2[1:-1] = ps[2:] - 2.0 * ps[1:-1] + ps[:-2]

    # --- 谷值權重（min–max 正規化，對齊論文）---
    d2 = np.maximum(d2, 0.0)          # 保留谷（正曲率）
    d2_min, d2_max = d2.min(), d2.max()
    rng = max(eps, d2_max - d2_min)
    wv = (d2 - d2_min) / rng        # 線性縮放到 [0,1]

    # --- Otsu 的類間方差 σ_b^2(t)（用平滑後的 ps + bin 中心） ---
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(ps)          # 左側權重
    mu_cum = np.cumsum(ps * centers)   # 左側一階動量
    mu_T = (ps * centers).sum()     # 全域均值

    denom = omega * (1.0 - omega)                 # ω0 * ω1
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b2 = (mu_T * omega - mu_cum) ** 2 / np.maximum(denom, eps)
        sigma_b2[(denom <= 0) | ~np.isfinite(sigma_b2)] = 0.0

    # --- MVE 目標：δ(t) = σ_b^2(t) * w_v(t)（索引長度對齊：皆為 bins 長度） ---
    # 註：傳統實作常在 t ∈ [0, L-2]；這裡最後一點 denom=0 已被置零，不影響 argmax。
    score = sigma_b2 * wv

    # 只在內部 bin 搜尋（與二階差分定義域一致）
    score_valid = score.copy()
    score_valid[0]  = -np.inf
    score_valid[-1] = -np.inf
    t_star = int(np.argmax(score_valid))
    return float(centers[t_star])



# ------------------------------------------------------------
# 內部工具：解 Gamma 形狀參數 alpha 的加權 MLE 方程 (log(alpha) - psi(alpha) = s)
#   s = log( weighted_mean(x) ) - weighted_mean(log x)
#   參考：Gamma MLE 的標準推導；加權時把均值改成加權均值即可。
# ------------------------------------------------------------
def _solve_alpha_newton(s, alpha0=1.0, max_iter=100, tol=1e-8):
    a = max(alpha0, 1e-6)
    for _ in range(max_iter):
        f  = np.log(a) - digamma(a) - s
        fp = 1.0/a - polygamma(1, a)  # trigamma = polygamma(1,·)
        step = f / fp
        a_new = a - step
        # 防數值炸裂：限制步長與範圍
        if not np.isfinite(a_new) or a_new <= 1e-8:
            a_new = max(a * 0.5, 1e-6)
        if abs(a_new - a) < tol * (1.0 + abs(a)):
            a = a_new
            break
        a = a_new
    return float(max(a, 1e-6))

# ------------------------------------------------------------
# 內部工具：在已知樣本 x 與權重 w 下，計算 Gamma(a, scale=theta) 的加權 MLE
#   回傳 (alpha, theta)
# ------------------------------------------------------------
def _weighted_gamma_mle(x, w, eps=1e-12):
    W = np.sum(w)
    if W <= 0:
        return 1.0, max(np.mean(x), eps)

    x = np.clip(x, eps, None)
    # 加權均值與加權 log 均值
    m     = np.sum(w * x) / W
    m_log = np.sum(w * np.log(x)) / W
    s = np.log(m) - m_log

    # 形狀參數 alpha：Newton 法
    alpha = _solve_alpha_newton(s, alpha0=max(m**2 / max(np.var(x), eps), 1.0))
    theta = m / alpha
    theta = max(theta, eps)
    return alpha, theta

# ------------------------------------------------------------
# EM：兩成份 Gamma 混合 (1D)，回傳 (pi, alpha, theta)，依均值從小到大排序
# ------------------------------------------------------------
def _fit_gamma_mixture_em(x, max_iter=500, tol=1e-6, random_state=0, init='gmm', eps=1e-12):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = np.clip(x, eps, None)  # 避免 log(0)

    N = x.size
    if N == 0:
        raise ValueError("Empty data for mixture fit.")

    # ---- 初始化 responsibilities ----
    # 預設用 sklearn GMM 幫你初值；若沒裝 sklearn，就用分位數分割
    # 先用標準的 Gaussian Mixture Model (GMM) 估出兩成份的混合權重與均值/標準差，再把這些參數當作最大概似擬合的起始值；若最後選的是 Gamma 分布，就把 GMM 的初值轉換成 Gamma 的 (shape, scale) 來開始擬合
    # 𝑟∈𝑅^(𝑁×2)，接著你的 EM 就用這個 𝑟去做 Gamma 的加權

    r = None
    if init == 'gmm':
        try:
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=2, covariance_type='diag',
                                 reg_covar=1e-6, random_state=random_state)
            gm.fit(x.reshape(-1,1))
            r = gm.predict_proba(x.reshape(-1,1))  # (N,2)
        except Exception:
            r = None
    if r is None:
        # 簡單二分：用中位數切；加點噪聲避免全 0/1
        m = np.median(x)
        r = np.zeros((N, 2), dtype=np.float64)
        r[:, 0] = (x <= m).astype(np.float64)
        r[:, 1] = 1.0 - r[:, 0]
        r += 1e-3 * rng.random(r.shape)
        r /= r.sum(axis=1, keepdims=True)

    # ---- EM 迭代 ----
    ll_prev = -np.inf
    for _ in range(max_iter):
        # M-step：更新 pi, (alpha,theta)
        pi = r.mean(axis=0)  # (2,)

        # 各成份的加權 MLE
        a, th = [], []
        for k in range(2):
            alpha_k, theta_k = _weighted_gamma_mle(x, r[:, k], eps)
            a.append(alpha_k); th.append(theta_k)
        a = np.array(a); th = np.array(th)

        # E-step：計算責任度
        # pdf_k(x) = GammaPDF(x; alpha_k, scale=theta_k)
        pdf0 = sp_gamma.pdf(x, a=a[0], scale=th[0])
        pdf1 = sp_gamma.pdf(x, a=a[1], scale=th[1])
        mix0 = pi[0] * pdf0
        mix1 = pi[1] * pdf1
        denom = np.maximum(mix0 + mix1, eps)
        r[:, 0] = mix0 / denom
        r[:, 1] = mix1 / denom

        # 收斂檢查：對數似然
        ll = np.sum(np.log(denom))
        if np.isfinite(ll) and (ll - ll_prev) < tol * (1.0 + abs(ll_prev)):
            break
        ll_prev = ll

    # 以均值排序（群1 = 小均值，群2 = 大均值）
    means = a * th
    order = np.argsort(means)
    pi, a, th = pi[order], a[order], th[order]
    return pi, a, th

# ------------------------------------------------------------
# 主函式：Gamma–Gamma 混合 + 掃描 t 以最大化 (SEN+SPC)/2
#   values：一維樣本；建議是 distance-to-nearest（論文定義）
#   若有 bins，會用 bins 的範圍來掃描門檻，便於與直方圖對齊
#   回傳：thr, params(dict)
# ------------------------------------------------------------
def gamma_mixture_threshold(values, bins=256, data_range=None,
                            grid=4096, random_state=0, eps=1e-12) -> tuple[float, dict] | tuple[None, None]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None

    # 掃描區間：盡量跟你的繪圖一致
    if isinstance(bins, (list, tuple, np.ndarray)):
        xmin, xmax = float(np.min(bins)), float(np.max(bins))
    else:
        xmin = float(np.min(v) if data_range is None else data_range[0])
        xmax = float(np.max(v) if data_range is None else data_range[1])

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
        xmin, xmax = float(np.min(v)), float(np.max(v))
        if xmin >= xmax:
            return float(xmin), None

    # 擬合 Gamma–Gamma 混合（兩成份）
    pi, a, th = _fit_gamma_mixture_em(v, random_state=random_state)

    # 定義兩類（依均值大小）：class 0 = 小距離（同群/克隆），class 1 = 大距離（不同群）
    # SEN(t) = P(class0 | x<t) = F0(t)；SPC(t) = P(class1 | x>=t) = 1 - F1(t)
    xs = np.linspace(xmin, xmax, int(grid))
    F0 = sp_gamma.cdf(xs, a=a[0], scale=th[0])
    F1 = sp_gamma.cdf(xs, a=a[1], scale=th[1])
    SEN = F0
    SPC = 1.0 - F1
    BA  = 0.5 * (SEN + SPC)  # Balanced Accuracy = (SEN+SPC)/2

    idx = int(np.argmax(BA))
    thr = float(xs[idx])

    params = {
        "pi": pi,                 # 混合權重（未用於 BA；BA 是不依先驗的平衡指標）
        "alpha": a, "theta": th,  # 兩個 Gamma 成份參數（依均值排序）
        "balanced_accuracy": float(BA[idx]),
        "grid_x": xs              # 若要畫曲線可用
    }
    return thr, params
