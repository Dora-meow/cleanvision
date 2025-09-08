import hashlib
import multiprocessing
from typing import Any, Dict, List, Optional, Union, Tuple
import os, gc, tempfile

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors #mode2
from scipy.sparse import coo_matrix #mode2
from scipy.sparse.csgraph import connected_components #mode2
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA #mode5

# NEW: for mode=8
from scipy.spatial.distance import pdist  # condensed distance for fastcluster
# 如果你原本沒有，補一下（mode=8 會用到）
from scipy.cluster.hierarchy import fcluster

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.issue_managers import register_issue_manager, IssueType
from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import SETS, DUPLICATE, MAX_PROCS
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname


def get_hash(image: Image.Image, params: Dict[str, Any]) -> str:
    hash_type, hash_size = params["hash_type"], params.get("hash_size", None)
    if hash_type == "md5":
        pixels = np.asarray(image)
        return hashlib.md5(pixels.tobytes()).hexdigest()
    elif hash_type == "whash":
        return str(imagehash.whash(image, hash_size=hash_size))
    elif hash_type == "phash":
        return str(imagehash.phash(image, hash_size=hash_size))
    elif hash_type == "ahash":
        return str(imagehash.average_hash(image, hash_size=hash_size))
    elif hash_type == "dhash":
        return str(imagehash.dhash(image, hash_size=hash_size))
    elif hash_type == "chash":
        return str(imagehash.colorhash(image, binbits=hash_size))
    else:
        raise ValueError("Hash type not supported")


def compute_hash(
    index: int,
    dataset: Dataset,
    to_compute: List[str],
    params: Dict[str, Any],
) -> Dict[str, Union[str, int]]:
    image = dataset[index]
    result: Dict[str, Union[str, int]] = {"index": index}
    if image:
        for issue_type in to_compute:
            result[issue_type] = get_hash(image, params[issue_type])
    return result


def compute_hash_wrapper(args: Dict[str, Any]) -> Dict[str, Union[str, int]]:
    return compute_hash(**args)


@register_issue_manager(DUPLICATE)
class DuplicateIssueManager(IssueManager):
    issue_name: str = DUPLICATE
    visualization: str = "image_sets"

    def __init__(self) -> None:
        super().__init__()
        self.issue_types: List[str] = []
        self.params = self.get_default_params()

    def get_default_params(self) -> Dict[str, Any]:
        return {
            IssueType.EXACT_DUPLICATES.value: {"hash_type": "md5"},
            IssueType.NEAR_DUPLICATES.value: {"hash_type": "phash", "hash_size": 8, 
                                              "mode": 0, #預設使用原本分群方式
                                              "min_cluster_size": 2,  #mode=1才會用到
                                              "hamming_dis": 10,     # for mode=2（單位：bit）
                                              "linkage": "average",  # for mode=3：'single' | 'complete' | 'average'
                                              "block_bits": 12,      # mode=4 用，前多少 bits 分桶
                                              "min_block_size_for_hier": 8,  # for mode=4: 桶內少於這個量，可退化成 k=1
                                              
                                              # 👇 mode=5（PCA）相關
                                              "dim_method": "pca",       # 可選: "pca" / "mca" / "famd"
                                              "pca_components": 20,
                                              "pca_whiten": True,        # 讓各主成分單位方差，距離更穩定
                                              "pca_metric": "euclidean", # PCA 後用的距離
                                              "dim_metric": "euclidean", # 新參數名（優先使用）
                                              }, 
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        for issue_type in self.params:
            non_none_params = {
                k: v for k, v in params.get(issue_type, {}).items() if v is not None
            }
            self.params[issue_type] = {**self.params[issue_type], **non_none_params}
            print(self.params)

    def _get_issue_types_to_compute(
        self, issue_types: List[str], imagelab_info: Dict[str, Any]
    ) -> List[str]:
        """Gets issue type for which computation needs to be done

        Only exact duplicate results are reused.
        Near duplicate hash computation is always done if specified in issue_types
        to reduce code complexity as it is parametric.

        Parameters
        ----------
        imagelab_info

        Returns
        -------
        to_compute : list
                     List of issue_types to run computation for
        """
        to_compute = []
        if SETS not in imagelab_info.get(IssueType.EXACT_DUPLICATES.value, {}):
            to_compute.append(IssueType.EXACT_DUPLICATES.value)
        if IssueType.NEAR_DUPLICATES.value in issue_types:
            to_compute.append(IssueType.NEAR_DUPLICATES.value)
        return to_compute

    def find_issues(
        self,
        *,
        params: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dataset] = None,
        imagelab_info: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().find_issues(**kwargs)
        assert params is not None
        assert imagelab_info is not None
        assert dataset is not None
        assert n_jobs is not None

        self.issue_types = list(params.keys())
        self.update_params(params)

        to_compute = self._get_issue_types_to_compute(self.issue_types, imagelab_info)
        issue_type_hash_mapping: Dict[str, Any] = {
            issue_type: {} for issue_type in to_compute
        }

        results: List[Dict[str, Union[str, int]]] = []
        if n_jobs == 1:
            for idx in tqdm(
                dataset.index, leave=verbose, desc="Computing hashes", smoothing=0
            ):
                results.append(compute_hash(idx, dataset, to_compute, self.params))
        else:
            args = [
                {
                    "index": idx,
                    "dataset": dataset,
                    "to_compute": to_compute,
                    "params": self.params,
                }
                for idx in dataset.index
            ]
            chunksize = max(1, len(args) // MAX_PROCS)
            with multiprocessing.Pool(n_jobs) as p:
                results = list(
                    tqdm(
                        p.imap_unordered(
                            compute_hash_wrapper, args, chunksize=chunksize
                        ),
                        total=len(dataset),
                        leave=verbose,
                        desc="Computing hashes",
                        smoothing=0,
                    )
                )

            results = sorted(results, key=lambda r: r["index"])

        # 根據每個 issue_type 的 output_hash 設定輸出 CSV
        for issue_type in to_compute:
            issue_params = self.params.get(issue_type, {})
            if issue_params.get("output_hash", False):
                rows = []
                for r in results:
                    if issue_type in r:
                        rows.append({
                            "index": r["index"],
                            "hash": r[issue_type]
                        })
                df = pd.DataFrame(rows)
                hash_type = issue_params.get("hash_type", "unknown")
                csv_name = f"{issue_type}_{hash_type}_hashes.csv"
                df.to_csv(csv_name, index=False)
                print(f"[✔] Hash CSV saved: {csv_name}")

        for result in results:
            for issue_type in to_compute:
                hash_str = result.get(issue_type, None)
                if hash_str in issue_type_hash_mapping[issue_type]:
                    issue_type_hash_mapping[issue_type][hash_str].append(
                        result["index"]
                    )
                else:
                    issue_type_hash_mapping[issue_type][hash_str] = [result["index"]]

        self.issues = pd.DataFrame(index=dataset.index)
        self._update_info(self.issue_types, issue_type_hash_mapping, imagelab_info)
        self._update_issues()
        self._update_summary()

        return

    def _update_summary(self) -> None:
        summary_dict = {}
        for issue_type in self.issue_types:
            summary_dict[issue_type] = self._compute_summary(
                self.issues[get_is_issue_colname(issue_type)]
            )
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index()
        self.summary = self.summary.rename(columns={"index": "issue_type"})
        self.summary = self.summary.astype({"issue_type": str})

    def _update_issues(self) -> None:
        score_df = pd.DataFrame(index=self.issues.index)
        score_df[get_score_colname(IssueType.EXACT_DUPLICATES.value)] = np.ones(
            len(score_df)
        )
        score_df[get_score_colname(IssueType.NEAR_DUPLICATES.value)] = np.ones(
            len(score_df)
        )

        for issue_type in self.issue_types:
            score_col = get_score_colname(issue_type)
            for s in self.info[issue_type][SETS]:
                score = 1.0 / len(
                    s
                )  # will never be 0 because all images in this set are duplicated
                score_df.loc[s, score_col] = score

            self.issues = self.issues.join(score_df[[score_col]])
            self.issues[get_is_issue_colname(issue_type)] = [
                False if x == 1 else True for x in self.issues[score_col]
            ]

    def _update_info(
        self,
        issue_types: List[str],
        issue_type_hash_mapping: Dict[str, Any],
        imagelab_info: Dict[str, Any],
    ) -> None:
        num_sets = "num_sets"
        if IssueType.EXACT_DUPLICATES.value in issue_type_hash_mapping:
            self.info[IssueType.EXACT_DUPLICATES.value] = {
                SETS: self._get_duplicate_sets(
                    issue_type_hash_mapping[IssueType.EXACT_DUPLICATES.value]
                )
            }
        else:
            self.info[IssueType.EXACT_DUPLICATES.value] = {
                SETS: imagelab_info[IssueType.EXACT_DUPLICATES.value][SETS]
            }
        self.info[IssueType.EXACT_DUPLICATES.value][num_sets] = len(
            self.info[IssueType.EXACT_DUPLICATES.value][SETS]
        )

        if IssueType.NEAR_DUPLICATES.value in issue_types:
            nd_params = self.params[IssueType.NEAR_DUPLICATES.value]
            mode = nd_params.get("mode", 0)
            print('mode: ', mode)
            print('nd_params: ', nd_params)
            if mode == 1:
                sets = self._cluster_near_duplicates_with_hamming(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 2:
                sets = self._cluster_near_duplicates_by_threshold(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 3: #階層式分群(省空間)
                sets = self._cluster_near_duplicates_hier_auto(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 4: #階層式分群+分桶
                sets = self._cluster_near_duplicates_blocked_hier_auto(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 5: #pca+階層式分群
                sets = self._cluster_near_duplicates_hier_pca_auto(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 6:  # 階層式分群+分桶+合併
                sets = self._cluster_near_duplicates_blocked_hier_auto_mode6(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 7:  # 階層式分群(原始版：全量 Hamming + 階層式 + 自動群數)
                sets = self._cluster_near_duplicates_hier_auto_ori(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 8:  # fastcluster
                sets = self._cluster_near_duplicates_hier_auto_fastcluster(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            else:
                print('----執行cleanvision原方法----')
                sets = self._get_duplicate_sets(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value]
                )

            self.info[IssueType.NEAR_DUPLICATES.value] = {SETS: sets}
            self._remove_exact_duplicates_from_near()
            self.info[IssueType.NEAR_DUPLICATES.value][num_sets] = len(
                self.info[IssueType.NEAR_DUPLICATES.value][SETS]
            )

    def _remove_exact_duplicates_from_near(self) -> None:
        updated_sets = []
        for s in self.info[IssueType.NEAR_DUPLICATES.value][SETS]:
            if s not in self.info[IssueType.EXACT_DUPLICATES.value][SETS]:
                updated_sets.append(s)
        self.info[IssueType.NEAR_DUPLICATES.value][SETS] = updated_sets

    @staticmethod
    def _get_duplicate_sets(hash_image_mapping: Dict[str, str]) -> List[str]:
        duplicate_sets = []
        for hash, images in hash_image_mapping.items():
            if len(images) > 1:
                duplicate_sets.append(images)
        return duplicate_sets
    
    # 將 hash 轉為 bit 向量
    @staticmethod
    def hex_to_bits(h: str) -> np.ndarray:
        return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))

    def _cluster_near_duplicates_with_hamming(
        self, hash_image_mapping: Dict[str, List[int]],
        params: Dict[str, Any]
    ) -> List[List[int]]:

        min_cluster_size = params.get("min_cluster_size", 2)

        # Hashes 是唯一值，對應到多張圖
        unique_hashes = list(hash_image_mapping.keys())
        index_mapping = {i: hash_image_mapping[h] for i, h in enumerate(unique_hashes)}

        bit_matrix = np.array([self.hex_to_bits(h) for h in unique_hashes])

        # 分群
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="hamming")
        labels = clusterer.fit_predict(bit_matrix)

        # 建立群組，將相同 cluster 的圖片合併
        group_map = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            group_map.setdefault(label, []).extend(index_mapping[i])

        return [imgs for imgs in group_map.values() if len(imgs) > 1]

    def _cluster_near_duplicates_by_threshold(
        self,
        hash_image_mapping: Dict[str, List[int]], #{ hash值字串 : [圖片index列表] }
        params: Dict[str, Any],
    ) -> List[List[int]]:
        hamming_dis = int(params.get("hamming_dis", 10))

        unique_hashes = list(hash_image_mapping.keys()) #取所有不同的 hash 值（去掉重複的）
        if not unique_hashes:
            return []

        # 唯一 hash 的 row index -> 實際圖片 indices
        idx_map = {i: hash_image_mapping[h] for i, h in enumerate(unique_hashes)}

        bit_matrix = np.array([self.hex_to_bits(h) for h in unique_hashes], dtype=np.uint8)
        n, n_bits = bit_matrix.shape
        radius = hamming_dis / float(n_bits)  # sklearn 的 hamming 是 0~1

        # 半徑鄰居圖（注意：這步對大 n 可能較慢，必要時加分桶/blocking）
        nn = NearestNeighbors(metric="hamming", radius=radius, n_jobs=-1)
        nn.fit(bit_matrix)
        neigh_inds = nn.radius_neighbors(bit_matrix, return_distance=False)

        # 構建無向圖的邊（只保留 i<j）
        rows, cols = [], []
        for i, neigh in enumerate(neigh_inds):
            for j in neigh:
                if j == i: 
                    continue
                if j > i:
                    rows.append(i); cols.append(j)
                else:
                    rows.append(j); cols.append(i)

        if not rows:
            return []

        data = np.ones(len(rows), dtype=np.uint8)
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))
        graph = graph + graph.T  # 對稱

        # 連通分量 = 一個群
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        buckets: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            buckets.setdefault(lbl, []).extend(idx_map[i])

        # 只保留大小 > 1 的群
        return [imgs for imgs in buckets.values() if len(imgs) > 1]


    def _cluster_near_duplicates_hier_auto(
            self,
            hash_image_mapping: Dict[str, List[int]],  # { hash字串 : [圖片index列表] }
            params: Dict[str, Any],
        ) -> List[List[int]]:

        linkage = params.get("linkage", "average")
        # 分批大小：控制一次計算多少列的距離，依 RAM 調整
        row_block = int(params.get("row_block", 2048))
        # 距離矩陣精度：float16 最省（亦可改 'float32'）
        dist_dtype = params.get("dist_dtype", "float16")

        unique_hashes = list(hash_image_mapping.keys())
        if not unique_hashes:
            return []

        n = len(unique_hashes)

        # ---- 1) 十六進位打包成位元組（不展開成 0/1） ----
        max_hex_len = max(len(h.strip()) for h in unique_hashes)
        bit_len = max_hex_len * 4
        nbytes = (bit_len + 7) // 8  # 每個樣本的位元組數

        # 將 hex 轉成固定長度的 uint8 bytes（大端）
        def hex_to_packed_bytes(hex_str: str, nbytes: int) -> np.ndarray:
            v = int(hex_str, 16)
            out = np.zeros(nbytes, dtype=np.uint8)
            for j in range(nbytes):
                out[nbytes - 1 - j] = v & 0xFF
                v >>= 8
            return out

        codes = np.empty((n, nbytes), dtype=np.uint8)
        for i, h in enumerate(unique_hashes):
            codes[i] = hex_to_packed_bytes(h, nbytes)

        # i -> 對應圖片索引清單
        idx_map = {i: hash_image_mapping[h] for i, h in enumerate(unique_hashes)}

        # ---- 2) 用 memmap 建立距離矩陣（對稱, 對角=0），避免把 n×n 放在 RAM ----
        # 值域是 Hamming 比例 (0~1)，用 float16 足夠
        tmpdir = params.get("tmpdir", None) or tempfile.gettempdir()
        dist_path = os.path.join(tmpdir, "hamming_dist_mm.dat")
        dist_mm = np.memmap(dist_path, dtype=dist_dtype, mode="w+", shape=(n, n))

        # popcount 查表（0..255）
        popcnt = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

        def hamming_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            a: (p, nbytes) uint8, b: (q, nbytes) uint8
            return (p, q) uint16 的「bit 數」距離
            """
            x = np.bitwise_xor(a[:, None, :], b[None, :, :])  # (p, q, nbytes)
            # 查表逐位元組 popcount，再沿最後一維加總
            return popcnt[x].sum(axis=2, dtype=np.uint16)

        # 分批填上三角矩陣，並鏡射到另一半；比例= bit_count / bit_len
        for i0 in range(0, n, row_block):
            a = codes[i0:i0 + row_block]  # (p, nbytes)
            for j0 in range(0, n, row_block):
                b = codes[j0:j0 + row_block]  # (q, nbytes)
                d_bits = hamming_block(a, b)  # (p, q) uint16
                # 轉成比例（float16/32）
                d = (d_bits.astype(np.float32) / float(bit_len)).astype(dist_mm.dtype, copy=False)

                # 寫入 memmap
                p, q = d.shape
                dist_mm[i0:i0+p, j0:j0+q] = d
                # 幫助釋放暫存
                del d, d_bits, b
                gc.collect()

            # 對角區塊（含自身）：確保對角線為 0
            p = min(row_block, n - i0)
            dist_mm[i0:i0+p, i0:i0+p][np.diag_indices(p)] = 0.0
            del a
            gc.collect()

        # 對稱化（避免數值微差）
        # 只需上三角覆蓋下三角或反之；這裡簡單平均一次
        upper = np.triu(dist_mm)
        dist_mm[:] = upper + upper.T - np.diag(np.diag(dist_mm))
        del upper
        gc.collect()

        # ---- 3) 先建整棵樹以取得 distances_（每次合併距離） ----
        # sklearn 新版用 metric，舊版用 affinity；包一層相容
        def _make_agg(**kw):
            try:
                return AgglomerativeClustering(metric="precomputed", linkage=linkage, **kw)
            except TypeError:
                return AgglomerativeClustering(affinity="precomputed", linkage=linkage, **kw)

        tree = _make_agg(distance_threshold=0.0, n_clusters=None, compute_distances=True).fit(dist_mm)
        d = getattr(tree, "distances_", None)
        if d is None or len(d) == 0:
            # 清理再回傳空
            del dist_mm, codes
            gc.collect()
            try:
                os.remove(dist_path)
            except Exception:
                pass
            return []

        # ---- 4) 最大跳躍法自動決 k ----
        '''
        階層式分群（Hierarchical Clustering）中，我們會從每個樣本一開始單獨成一群，
        然後逐步把距離最近的群合併，直到只剩下一群。
        整個過程會記錄下每一次「合併」的距離（例如 Hamming 距離或歐氏距離），把距離排成遞增序列。
        計算相鄰距離的差值，取差值最大的那個合併當群數
        ex：
        每次合併的距離=[0.05, 0.06, 0.08, 0.12, 0.95, 1.02]
        0.12~0.95差距最大 -> 這次合併把不像的併在一起了 -> 找到適合分群的地方 -> k=2
        '''
        d_sorted = np.sort(d)
        if len(d_sorted) <= 1:
            k = 1
        else:
            gaps = np.diff(d_sorted)
            if len(gaps) == 0:
                k = max(1, min(n, 2))
            else:
                j = int(np.argmax(gaps))
                cut = 0.5 * (d_sorted[j] + d_sorted[j+1])
                merges_below = int(np.sum(d <= cut))
                k = int(max(1, n - merges_below))

        # 釋放不需要的暫存
        del d, d_sorted, tree
        gc.collect()

        # ---- 5) 依 k 重新分群取得 labels（直接重用 memmap 距離） ----
        model = _make_agg(n_clusters=k, distance_threshold=None, compute_distances=False).fit(dist_mm)
        labels = model.labels_.copy()  # 複製後可釋放 model
        del model
        gc.collect()

        # 我們已經用完距離矩陣，釋放並刪檔
        del dist_mm, codes
        gc.collect()
        try:
            os.remove(dist_path)
        except Exception:
            pass

        # ---- 6) 聚合群結果 ----
        buckets: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            buckets.setdefault(lbl, []).extend(idx_map[i])

        return [imgs for imgs in buckets.values() if len(imgs) > 1]

    def _cluster_near_duplicates_blocked_hier_auto(
        self,
        hash_image_mapping: Dict[str, List[int]],
        params: Dict[str, Any],
    ) -> List[List[int]]:
        """
        mode=4: prefix blocking + 桶內 Agglomerative（自動K，且加群內直徑上限）;
                最後濾掉太小的群。
        """
        linkage = params.get("linkage", "average")            # 建議先試 'complete' 更保守
        block_bits = int(params.get("block_bits", 12))
        min_block = int(params.get("min_block_size_for_hier", 8))
        min_set_size = int(params.get("min_set_size", 2))     # 最小群大小
        max_intra_bits = params.get("max_intra_hamming_bits", None)  # 群內直徑上限（bit）

        unique_hashes = list(hash_image_mapping.keys())
        if not unique_hashes:
            return []

        hex_list: List[str] = unique_hashes
        row_to_img_indices: Dict[int, List[int]] = {
            i: hash_image_mapping[h] for i, h in enumerate(hex_list)
        }

        # 分桶（用十六進位前 prefix_len 字元 ≈ block_bits//4 bits）
        buckets: Dict[str, List[int]] = {}
        if block_bits > 0:
            prefix_len = max(1, block_bits // 4) #1 個十六進位字元 = 4 bits
            for i, h in enumerate(hex_list):
                key = h[:prefix_len] if len(h) >= prefix_len else h #用prefix_len當桶的key
                buckets.setdefault(key, []).append(i)
        else:
            buckets = {"__all__": list(range(len(hex_list)))} #block_bits <= 0，就不分桶，全部放在 "__all__" 一桶

        def auto_k_and_cut(dist_mat: np.ndarray):
            """回傳 (k, cut_ratio)。cut_ratio ∈ [0,1] 為自動判斷的距離門檻（Hamming 比例）。"""
            tree = AgglomerativeClustering(
                metric="precomputed",
                linkage=linkage,
                distance_threshold=0.0,
                n_clusters=None,
                compute_distances=True,
            ).fit(dist_mat)
            d = tree.distances_
            if len(d) == 0:
                return 1, 0.0
            d_sorted = np.sort(d)
            gaps = np.diff(d_sorted)
            if len(gaps) == 0:
                k = 2 if dist_mat.shape[0] >= 2 else 1
                cut = float(d_sorted[0]) if len(d_sorted) > 0 else 0.0
                return k, cut
            j = int(np.argmax(gaps))
            cut = 0.5 * (d_sorted[j] + d_sorted[j + 1])        # 自動切點（0~1）
            merges_below = np.sum(d <= cut)
            k = int(max(1, dist_mat.shape[0] - merges_below))
            return k, float(cut)

        all_sets: List[List[int]] = []

        for _, rows in buckets.items():
            n = len(rows)
            if n < 2:
                continue

            bit_matrix = np.array([self.hex_to_bits(hex_list[i]) for i in rows], dtype=np.uint8)
            n_bits = bit_matrix.shape[1]
            dist = pairwise_distances(bit_matrix, metric="hamming")  # 0~1

            if n < min_block:
                # 小桶：直接用嚴格的直徑限制（若有設定），否則每點自成群（不會輸出 size<2）
                if max_intra_bits is None:
                    # 退化：用連通分量的想法（每點不會合併，後面會過濾 size<2）
                    labels = np.arange(n)
                else:
                    thr = float(max_intra_bits) / float(n_bits)
                    model = AgglomerativeClustering(
                        metric="precomputed", linkage=linkage,
                        distance_threshold=thr, n_clusters=None
                    ).fit(dist)
                    labels = model.labels_
            else:
                # 自動決定切點 & k
                k_auto, cut_ratio = auto_k_and_cut(dist)

                # 若有直徑上限，用更嚴格者：min(自動切點, 上限)
                if max_intra_bits is not None:
                    cap_ratio = float(max_intra_bits) / float(n_bits)
                    cut_ratio = min(cut_ratio, cap_ratio)

                    # 用 distance_threshold 直接切（更穩，避免二次估 k 誤差）
                    model = AgglomerativeClustering(
                        metric="precomputed", linkage=linkage,
                        distance_threshold=cut_ratio, n_clusters=None
                    ).fit(dist)
                    labels = model.labels_
                else:
                    # 沒有直徑上限，就用自動 k
                    model = AgglomerativeClustering(
                        metric="precomputed", linkage=linkage,
                        n_clusters=int(k_auto)
                    ).fit(dist)
                    labels = model.labels_

            # 回填圖片 indices；濾掉太小的群
            bucket_map: Dict[int, List[int]] = {}
            for local_i, lbl in enumerate(labels):
                global_row = rows[local_i]
                bucket_map.setdefault(lbl, []).extend(row_to_img_indices[global_row])

            for imgs in bucket_map.values():
                if len(imgs) >= min_set_size:            # 👈 濾小群
                    all_sets.append(imgs)

        return all_sets

    def _cluster_near_duplicates_hier_pca_auto(
            self,
            hash_image_mapping: Dict[str, List[int]],
            params: Dict[str, Any],
        ) -> List[List[int]]:
        """
        mode=5: pHash bit -> (PCA | MCA | FAMD | logical_pca ) -> pairwise distances (euclidean by default) ->
                AgglomerativeClustering（最大跳躍法自動決定切點/群數）。
        * PCA：連續空間；適合把 bit 當數值信號
        * MCA：類別空間（bit 視為類別 0/1），需要 prince；失敗則退回 PCA
        * FAMD：數值+類別混合；在全類別時 ~ 等同 MCA
        參數：
        - linkage: 'average' | 'complete' | 'single' ...
        - dim_method: 'pca' | 'mca' | 'famd'（預設 pca）
        - pca_components: int（目標維度上限）
        - pca_whiten: bool（PCA 用）(每個特徵向量除以其標準差)
        - dim_metric: 距離度量（預設 'euclidean'）
        """

        linkage = params.get("linkage", "average")
        dim_method = str(params.get("dim_method", "pca")).lower()   # 'pca' | 'mca' | 'famd' | 'logistic_pca'
        n_comp = int(params.get("pca_components", 20))              # 與舊參數相容
        whiten = bool(params.get("pca_whiten", True))
        dim_metric = params.get("dim_metric", params.get("pca_metric", "euclidean"))

        unique_hashes = list(hash_image_mapping.keys())
        if not unique_hashes:
            return []

        # 唯一 hash 列表與 row->images 對應
        hex_list: List[str] = unique_hashes
        row_to_img_indices: Dict[int, List[int]] = {
            i: hash_image_mapping[h] for i, h in enumerate(hex_list)
        }

        # 1) pHash hex -> bit 向量（0/1）
        bit_matrix = np.array([self.hex_to_bits(h) for h in hex_list], dtype=np.uint8)
        n_samples, n_bits = bit_matrix.shape
        n_components = max(1, min(n_comp, n_bits, n_samples - 1)) if n_samples > 1 else 1

        # 2) 依據 dim_method 做降維
        if dim_method == "pca":
            bit_f = bit_matrix.astype(np.float32)
            pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
            X = pca.fit_transform(bit_f)

        elif dim_method == "logistic_pca":
            try:
                # 專門處理 0/1 資料的 Logistic PCA（機率式），
                # 計算較慢、可能不收斂，但在二元資料上理論上更貼近。
                from logisticpca import LogisticPCA

                # 將 bit 矩陣轉成 float64（套件多半以雙精度運算較穩定）
                bit_f = bit_matrix.astype(np.float64, copy=False)

                # 決定實際要用的降維維度（不能超過樣本數-1，也不能超過特徵數）
                n_eff = max(1, min(n_components, bit_f.shape[1], bit_f.shape[0] - 1))
                m_val   = 5  # 多數實作的預設截斷強度

                # 僅傳必要與已知的參數，降低與舊版/新版套件簽名不合的風險
                lpca = LogisticPCA(n_features=bit_matrix.shape[1], n_components=n_eff, m=m_val)

                # 偵測是否為 torch 版（模型內含 Parameter）
                use_torch = False
                try:
                    import torch, torch.nn as nn
                    use_torch = any(isinstance(getattr(lpca, name), nn.Parameter)
                                    for name in dir(lpca))
                except Exception:
                    use_torch = False

                # 餵資料：torch 版就給 tensor；否則給 numpy
                if use_torch:
                    import torch
                    X_in = torch.as_tensor(bit_f, dtype=torch.float32)
                else:
                    X_in = bit_f

                # fit / transform 相容處理
                if hasattr(lpca, "fit_transform"):
                    X_out = lpca.fit_transform(X_in)
                else:
                    lpca.fit(X_in)
                    if hasattr(lpca, "transform"):
                        X_out = lpca.transform(X_in)
                    elif hasattr(lpca, "scores_"):
                        X_out = lpca.scores_
                    else:
                        raise AttributeError("LogisticPCA fitted but no transform/scores_")

                # 轉回 numpy float32
                if use_torch and hasattr(X_out, "detach"):
                    X = X_out.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    X = np.asarray(X_out, dtype=np.float32)
                    
            except Exception as e:
                # 若無法匯入或訓練失敗（不收斂/數值問題），退回輕量近似：logical_pca
                # （先做 Bernoulli 標準化，再跑一般 PCA）
                print(f"[WARN] LogisticPCA 不可用或失敗，改用 logical_pca。原因: {e}")

                # Bernoulli 標準化：對每一維用 p(1-p) 做標準化，p=0/1 無變異者移除
                eps = 1e-12
                p_hat = bit_matrix.mean(axis=0, dtype=np.float64) # 每一維的 1 的比例
                var = p_hat * (1.0 - p_hat)                           # Bernoulli 變異 p(1-p)
                keep = var > eps                              # 只保留有變異的維度
                if not np.any(keep):
                    # 萬一所有 bit 都常數，退回普通 PCA（會得到零向量）
                    bit_f = bit_matrix.astype(np.float32)
                    pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
                    X = pca.fit_transform(bit_f)
                else:
                    Z = (bit_matrix[:, keep] - p_hat[keep]) / np.sqrt(var[keep])
                    Z = Z.astype(np.float32, copy=False)
                    pca = PCA(n_components=min(n_components, Z.shape[1]), whiten=whiten, random_state=0)
                    X = pca.fit_transform(Z)

        elif dim_method in ("mca", "famd"):
            try:
                import pandas as pd
                import prince  # 需已安裝
                # 將 0/1 當作類別特徵
                df = pd.DataFrame(bit_matrix).astype("category")
                if dim_method == "mca":
                    model = prince.MCA(
                        n_components=n_components,
                        n_iter=3,
                        copy=True,
                        random_state=0,
                    )
                else:  # 'famd'
                    model = prince.FAMD(
                        n_components=n_components,
                        n_iter=3,
                        copy=True,
                        random_state=0,
                    )
                X = model.fit(df).transform(df).to_numpy(dtype=np.float32)
            except Exception as e:
                print(f"[WARN] {dim_method.upper()} 需要 'prince' 或執行失敗，改用 PCA。原因: {e}")
                bit_f = bit_matrix.astype(np.float32)
                pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
                X = pca.fit_transform(bit_f)
        else:
            print(f"[WARN] 未知 dim_method='{dim_method}'，改用 PCA。")
            bit_f = bit_matrix.astype(np.float32)
            pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
            X = pca.fit_transform(bit_f)

        # 3) 降維空間 pairwise 距離
        dist = pairwise_distances(X, metric=dim_metric)

        # 4) 最大跳躍法：先建整棵樹抓 distances_，自動決 cut/k
        def auto_k_and_cut(dist_mat):
            tree = AgglomerativeClustering(
                metric="precomputed",
                linkage=linkage,
                distance_threshold=0.0,
                n_clusters=None,
                compute_distances=True,
            ).fit(dist_mat)
            d = tree.distances_
            if len(d) == 0:
                return 1, 0.0
            d_sorted = np.sort(d)
            gaps = np.diff(d_sorted)
            if len(gaps) == 0:
                k = 2 if dist_mat.shape[0] >= 2 else 1
                cut = float(d_sorted[0]) if len(d_sorted) > 0 else 0.0
                return k, cut
            j = int(np.argmax(gaps))
            cut = 0.5 * (d_sorted[j] + d_sorted[j + 1])
            merges_below = np.sum(d <= cut)
            k = int(max(1, dist_mat.shape[0] - merges_below))
            return k, float(cut)

        k_auto, cut = auto_k_and_cut(dist)

        # 5) 用 distance_threshold 直接切（通常比固定 k 更穩）
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            distance_threshold=cut,
            n_clusters=None,
        ).fit(dist)
        labels = model.labels_

        # 6) 回填到圖片索引；只保留大小>1
        lbl2imgs: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            lbl2imgs.setdefault(lbl, []).extend(row_to_img_indices[i])

        return [imgs for imgs in lbl2imgs.values() if len(imgs) > 1]

    def _cluster_near_duplicates_blocked_hier_auto_mode6(
        self,
        hash_image_mapping: Dict[str, List[int]],
        params: Dict[str, Any],
    ) -> List[List[int]]:
        """
        mode=6: prefix blocking + 桶內 Agglomerative（自動決定群數）+（可選）跨桶縫合。
        特點：
        - 先用 block_bits 切成桶（prefix blocking）
        - 桶內用 Agglomerative 聚類（自動 k）
        - 可選 stitching：跨桶縫合相似群（prefix 近似 + bit 向量 Hamming 距離）
        """

        # -----------------------------
        # 基本參數
        # -----------------------------
        linkage = params.get("linkage", "average")          # Agglomerative linkage 方法
        block_bits = int(params.get("block_bits", 12))       # prefix blocking 長度（bit 數）
        min_block = int(params.get("min_block_size_for_hier", 8))  # 桶內最少樣本才聚類

        # stitching 相關參數（跨桶縫合）
        stitching = bool(params.get("stitching", False))
        stitch_thr = int(params.get("stitch_threshold", 8))          # 代表點 Hamming 門檻
        stitch_pref_h = int(params.get("stitch_prefix_hamming", 1))  # 舊版 prefix 鄰近（不再用）
        stitch_reps = int(params.get("stitch_sample_reps", 3))       # 每群取幾個代表點
        stitch_max_pairs = int(params.get("stitch_max_pairs", 200_000))  # 最多比對的群對數
        stitch_min_matches = int(params.get("stitch_min_matches", 2))    # 至少幾對代表點匹配才合併

        # -----------------------------
        # Union-Find（Disjoint Set Union, DSU）初始化
        # -----------------------------
        parent = {}
        def find(x):
            """找集合代表（帶路徑壓縮）"""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """合併兩個集合"""
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # -----------------------------
        # 資料準備
        # -----------------------------
        unique_hashes = list(hash_image_mapping.keys())
        if not unique_hashes:
            return []

        hex_list: List[str] = unique_hashes  # 所有 hash（十六進位字串）
        row_to_img_indices: Dict[int, List[int]] = {
            i: hash_image_mapping[h] for i, h in enumerate(hex_list)
        }

        # -----------------------------
        # prefix blocking（依前綴分桶）
        # -----------------------------
        buckets: Dict[str, List[int]] = {}
        if block_bits > 0:
            prefix_len = max(1, block_bits // 4)  # 每 4 bit 對應 1 hex 字元
            for i, h in enumerate(hex_list):
                key = h[:prefix_len] if len(h) >= prefix_len else h
                buckets.setdefault(key, []).append(i)
        else:
            buckets = {"__all__": list(range(len(hex_list)))}

        # -----------------------------
        # 自動決定 K（群數）
        # -----------------------------
        def auto_k_from_dist(dist_mat: np.ndarray) -> int:
            """
            根據距離矩陣找最佳切割點（最大間距法）
            """
            tree = AgglomerativeClustering(
                metric="precomputed",
                linkage=linkage,
                distance_threshold=0.0,
                n_clusters=None,
                compute_distances=True,
            ).fit(dist_mat)
            d = tree.distances_
            if len(d) == 0:
                return 1
            d_sorted = np.sort(d)
            gaps = np.diff(d_sorted)
            if len(gaps) == 0:
                return 2 if dist_mat.shape[0] >= 2 else 1
            j = int(np.argmax(gaps))
            cut = 0.5 * (d_sorted[j] + d_sorted[j + 1])
            merges_below = np.sum(d <= cut)
            k = int(max(1, dist_mat.shape[0] - merges_below))
            return k

        # -----------------------------
        # 初始化變數
        # -----------------------------
        all_sets: List[List[int]] = []   # 最終群組結果
        clusters_meta = []               # 儲存 stitching 所需中繼資料
        hex2bits: Dict[str, np.ndarray] = {}

        def get_bits(h: str) -> np.ndarray:
            """
            將十六進位 hash 轉成 bit 向量（一次轉換後快取）
            """
            if h not in hex2bits:
                hex2bits[h] = self.hex_to_bits(h)
            return hex2bits[h]

        # -----------------------------
        # 桶內處理
        # -----------------------------
        for key, rows in buckets.items():
            n = len(rows)
            if n < 2:
                continue

            # 轉 bit 向量
            bit_matrix = np.array([get_bits(hex_list[i]) for i in rows], dtype=np.uint8)

            # 桶內距離矩陣（Hamming）
            dist = pairwise_distances(bit_matrix, metric="hamming")

            # 決定群數
            k = 1 if n < min_block else auto_k_from_dist(dist)

            # 桶內 Agglomerative 聚類
            model = AgglomerativeClustering(
                metric="precomputed",
                linkage=linkage,
                n_clusters=k,
            ).fit(dist)
            labels = model.labels_

            # 分群結果
            lbl_to_local_rows: Dict[int, List[int]] = {}
            for local_i, lbl in enumerate(labels):
                lbl_to_local_rows.setdefault(lbl, []).append(local_i)

            for lbl, local_rows in lbl_to_local_rows.items():
                members: List[int] = []
                for local_i in local_rows:
                    global_row = rows[local_i]
                    members.extend(row_to_img_indices[global_row])

                if len(members) > 1:
                    all_sets.append(members)

                # stitching 代表點準備
                if stitching and len(local_rows) > 0:
                    member_rows = [rows[li] for li in local_rows]
                    mbits = np.array([get_bits(hex_list[r]) for r in member_rows], dtype=np.uint8)

                    # 找 medoid（總距離最小的點）
                    mdist = pairwise_distances(mbits, metric="hamming") * mbits.shape[1]
                    order = np.argsort(mdist.sum(axis=1))
                    rep_idx = order[: min(stitch_reps, len(order))]
                    reps_bits = mbits[rep_idx]

                    clusters_meta.append({
                        "prefix": key,
                        "members": members,
                        "reps_bits": reps_bits,
                    })

        if not stitching or len(clusters_meta) <= 1:
            return all_sets

        # ADDED: 這一段是「初始化 Union-Find 的 parent」，要放在
        #        桶內分群與收集 clusters_meta 完成之後、進入縫合流程之前。
        parent = {cid: cid for cid in range(len(clusters_meta))}

        # -----------------------------
        # 進階版 prefix 鄰近（bit 級）
        # -----------------------------
        pref_bits    = int(params.get("stitch_prefix_bits", block_bits))         # 前綴長度（bit 數）
        pref_bit_thr = int(params.get("stitch_prefix_hamming_bits", 0))          # 前綴允許 bit Hamming

        def _hex_prefix_to_bits(hex_prefix: str, num_bits: int) -> np.ndarray:
            """
            十六進位前綴 -> bit 向量，並裁到 num_bits
            """
            out = np.empty(len(hex_prefix) * 4, dtype=np.uint8)
            w = 0
            for ch in hex_prefix:
                v = int(ch, 16)
                out[w:w+4] = ((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1)
                w += 4
            out = out[:min(num_bits, w)]
            if out.size < num_bits:
                out = np.pad(out, (0, num_bits - out.size), constant_values=0)
            return out

        def prefix_hamming_bits(a_hex_prefix: str, b_hex_prefix: str) -> int:
            """
            計算兩個十六進位前綴的 bit 級 Hamming 距離
            """
            ab = _hex_prefix_to_bits(a_hex_prefix, pref_bits)
            bb = _hex_prefix_to_bits(b_hex_prefix,  pref_bits)
            return int(np.bitwise_xor(ab, bb).sum())

        # prefix -> cluster id
        from collections import defaultdict
        pref2cids = defaultdict(list)
        for cid, meta in enumerate(clusters_meta):
            pref2cids[meta["prefix"]].append(cid)

        keys = list(pref2cids.keys())
        near_keys = defaultdict(list)
        for i, ka in enumerate(keys):
            near_keys[ka].append(ka)  # 自己一定包含
            for kb in keys[i+1:]:
                if prefix_hamming_bits(ka, kb) <= pref_bit_thr:
                    near_keys[ka].append(kb)
                    near_keys[kb].append(ka)

        # -----------------------------
        # 跨桶縫合（代表點比對）
        # -----------------------------
        compared = 0
        for ka in keys:
            for kb in near_keys[ka]:
                for ca in pref2cids[ka]:
                    for cb in pref2cids[kb]:
                        if ca >= cb:
                            continue
                        if compared >= stitch_max_pairs:
                            break

                        # 代表點距離矩陣（bits）
                        A = clusters_meta[ca]["reps_bits"]
                        B = clusters_meta[cb]["reps_bits"]
                        xor = np.bitwise_xor(A[:, None, :], B[None, :, :])
                        dists = np.unpackbits(xor, axis=2).sum(axis=2)
                        min_d = int(dists.min())
                        close_pairs = int((dists <= stitch_thr).sum())

                        compared += 1
                        if (min_d <= stitch_thr) and (close_pairs >= stitch_min_matches):
                            union(ca, cb)
                if compared >= stitch_max_pairs:
                    break
            if compared >= stitch_max_pairs:
                break

        # -----------------------------
        # 合併 stitching 結果
        # -----------------------------
        comp2members: Dict[int, List[int]] = {}
        for cid, meta in enumerate(clusters_meta):
            r = find(cid)
            comp2members.setdefault(r, []).extend(meta["members"])

        stitched_sets = []
        for mems in comp2members.values():
            if len(mems) > 1:
                stitched_sets.append(sorted(set(mems)))

        all_sets.extend(stitched_sets)

        # 去重
        uniq = []
        seen = set()
        for s in all_sets:
            key = tuple(sorted(set(s)))
            if key not in seen:
                seen.add(key)
                uniq.append(list(key))
        return uniq

    def _cluster_near_duplicates_hier_auto_ori(
        self,
        hash_image_mapping: Dict[str, List[int]], #{ hash值字串 : [圖片index列表] }
        params: Dict[str, Any],
    ) -> List[List[int]]:
        """
        mode=7：原始版（不分桶、不做 memmap），直接以 bit Hamming 距離建整棵樹，
        用最大跳躍法自動決定群數，再回填為圖片索引集合。
        注意：記憶體/時間複雜度 O(n^2)，其中 n = 唯一 hash 個數。
        """
        linkage = params.get("linkage", "average")

        unique_hashes = list(hash_image_mapping.keys()) #所有不同的 hash 值（去掉重複的）
        if not unique_hashes:
            return []

        idx_map = {i: hash_image_mapping[h] for i, h in enumerate(unique_hashes)}

        bit_matrix = np.array([self.hex_to_bits(h) for h in unique_hashes], dtype=np.uint8)
        n, n_bits = bit_matrix.shape

        # O(n^2) 注意：n 是「唯一 hash 數」
        dist = pairwise_distances(bit_matrix, metric="hamming")  # 每對 hash 的 Hamming 距離比例 (0~1)

        # 先建整棵樹以取得 distances_（每次合併距離）
        tree = AgglomerativeClustering(
            metric="precomputed", #用自己的距離矩陣(沒hamming可選)
            linkage=linkage,
            distance_threshold=0.0, #(不切群
            n_clusters=None, #(不切群
            compute_distances=True, #可以讀取每次合併的距離 tree.distances_
        ).fit(dist)

        # 自動決定群數（最大跳躍法）
        # 在階層式合併的距離中，找到一次合併距離跳最遠的地方作為切點
        # 切點距離以下 → 保留為不同群
        # 切點距離以上 → 認為應該分開
        d = tree.distances_
        if len(d) == 0:
            return []

        d_sorted = np.sort(d)
        gaps = np.diff(d_sorted)
        if len(gaps) == 0:
            k = max(1, min(n, 2))
        else:
            j = int(np.argmax(gaps))
            cut = 0.5 * (d_sorted[j] + d_sorted[j+1])
            merges_below = np.sum(d <= cut) # 切點以下的合併數
            k = int(max(1, n - merges_below)) #自動算出來的群數

        # 依自動決定的 k 重新分群取得 labels
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            n_clusters=k,
        ).fit(dist)
        labels = model.labels_

        buckets: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            buckets.setdefault(lbl, []).extend(idx_map[i]) #把同一群的所有 hash 對應的圖片 index 合併

        return [imgs for imgs in buckets.values() if len(imgs) > 1] #保留群大小 > 1 的結果


    def _cluster_near_duplicates_hier_auto_fastcluster(
        self,
        hash_image_mapping: Dict[str, List[int]],
        params: Dict[str, Any],
    ) -> List[List[int]]:
        """
        mode=8：使用 fastcluster 進行階層式分群。
        與 mode=7 類似（全量 Hamming、O(n^2)），但以 fastcluster.linkage 加速。
        步驟：
          1) pHash 轉 bit 向量
          2) 用 pdist(bit_matrix, 'hamming') 得到 condensed 距離 D（0~1）
          3) fastcluster.linkage(D, method=linkage) 建樹
          4) 取 Z[:,2]（每次合併距離）用「最大跳躍法」自動決 k
          5) 用 fcluster 切群並回填為圖片索引
        注意：
          - linkage='ward' 需使用歐氏距離與原始樣本，不適合 Hamming 的 condensed D，故在此自動改為 'average'
        """
        try:
            import fastcluster
        except ImportError as e:
            raise ImportError(
                "mode=8 需要 fastcluster，請先安裝：pip install fastcluster"
            ) from e

        linkage = params.get("linkage", "average")
        if str(linkage).lower() == "ward":
            print("[WARN] mode=8 使用的是 Hamming condensed 距離，'ward' 不相容，改用 'average'")
            linkage = "average"

        # 蒐集唯一 hash 與回填對應
        unique_hashes = list(hash_image_mapping.keys())
        if not unique_hashes:
            return []

        idx_map = {i: hash_image_mapping[h] for i, h in enumerate(unique_hashes)}

        # 1) 轉 bit 矩陣（0/1）
        bit_matrix = np.array([self.hex_to_bits(h) for h in unique_hashes], dtype=np.uint8)
        n = bit_matrix.shape[0]
        if n < 2:
            return []

        # 2) condensed 距離（0~1 的 Hamming 比例）
        #    註：與 mode=7 的 square 距離矩陣不同，fastcluster 對非 Ward 方法更偏好 condensed 向量
        D = pdist(bit_matrix, metric="hamming")

        # 3) 建 linkage 樹
        Z = fastcluster.linkage(D, method=linkage)  # 形狀 (n-1, 4)，Z[:,2] 是每次合併距離

        # 4) 最大跳躍法決定k
        d = Z[:, 2]
        if d.size == 0: #n < 2 或無有效合併資料，直接無群可切
            return []

        d_sorted = np.sort(d) #把距離排序(越接近樹的根越大)
        gaps = np.diff(d_sorted) #算兩兩差距
        if gaps.size == 0: #n <= 2（只有 0 或 1 次合併）；或所有合併距離幾乎相等（無明顯「最大跳躍」)
            k = max(1, min(n, 2))
            labels = fcluster(Z, t=k, criterion="maxclust")
        else:
            j = int(np.argmax(gaps)) #j 是 最大 gap 的左端索引，代表「第 d_sorted[j] 與 d_sorted[j+1] 之間」的距離落差最大。
            cut = 0.5 * (d_sorted[j] + d_sorted[j + 1]) #取最大跳躍兩端的中點當切點
            merges_below = int((d <= cut).sum()) # 算在距離 ≤ cut 的高度下，發生多少次合併
            k = max(1, n - merges_below) 
            labels = fcluster(Z, t=k, criterion="maxclust")

        # 6) 聚合到圖片索引
        buckets: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            # fcluster 的標籤從 1 起算
            buckets.setdefault(int(lbl), []).extend(idx_map[i])

        # 只保留大小 > 1 的群
        return [imgs for imgs in buckets.values() if len(imgs) > 1]
