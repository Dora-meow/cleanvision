import hashlib
import multiprocessing
from typing import Any, Dict, List, Optional, Union

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
                                              "pca_components": 20,
                                              "pca_whiten": True,        # 讓各主成分單位方差，距離更穩定
                                              "pca_metric": "euclidean", # PCA 後用的距離
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
            elif mode == 3: #階層式分群
                sets = self._cluster_near_duplicates_hier_auto(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 4: #階層式分群+分桶
                sets = self._cluster_near_duplicates_blocked_hier_auto(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            elif mode == 5:
                sets = self._cluster_near_duplicates_hier_pca_auto(
                    issue_type_hash_mapping[IssueType.NEAR_DUPLICATES.value], nd_params
                )
            else:
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
        hash_image_mapping: Dict[str, List[int]], #{ hash值字串 : [圖片index列表] }
        params: Dict[str, Any],
    ) -> List[List[int]]:
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
            metric="precomputed",
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
        min_set_size = int(params.get("min_set_size", 2))     # 👈 新增：最小群大小
        max_intra_bits = params.get("max_intra_hamming_bits", None)  # 👈 新增：群內直徑上限（bit）

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
            prefix_len = max(1, block_bits // 4)
            for i, h in enumerate(hex_list):
                key = h[:prefix_len] if len(h) >= prefix_len else h
                buckets.setdefault(key, []).append(i)
        else:
            buckets = {"__all__": list(range(len(hex_list)))}

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
        mode=5: pHash bit -> PCA(20) -> pairwise distances (euclidean) ->
                AgglomerativeClustering，透過最大跳躍自動決定切點/群數。
        注意：這裡使用的是 PCA 空間的歐幾里得距離（不是 Hamming）。
        """
        linkage = params.get("linkage", "average")
        n_comp = int(params.get("pca_components", 20))
        whiten = bool(params.get("pca_whiten", True))
        pca_metric = params.get("pca_metric", "euclidean")

        unique_hashes = list(hash_image_mapping.keys())
        if not unique_hashes:
            return []

        # 唯一 hash 列表與 row->images 對應
        hex_list: List[str] = unique_hashes
        row_to_img_indices: Dict[int, List[int]] = {
            i: hash_image_mapping[h] for i, h in enumerate(hex_list)
        }

        # 1) pHash hex -> bit 向量（0/1），轉 float 以便 PCA
        bit_matrix = np.array([self.hex_to_bits(h) for h in hex_list], dtype=np.uint8).astype(np.float32)
        n_samples, n_bits = bit_matrix.shape

        # 安全設定 n_components
        n_components = max(1, min(n_comp, n_bits, n_samples - 1)) if n_samples > 1 else 1

        # 2) PCA 映射到低維
        pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
        X = pca.fit_transform(bit_matrix)  # shape: (n_samples, n_components)

        # 3) 計算 PCA 空間的 pairwise 距離（通常用 euclidean）
        dist = pairwise_distances(X, metric=pca_metric)

        # 4) 先建整棵樹抓 distances_，用最大跳躍決定切點/群數
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

        # 5) 用 distance_threshold 直接切（比用 n_clusters 更穩）
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
