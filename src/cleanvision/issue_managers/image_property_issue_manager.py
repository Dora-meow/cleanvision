import multiprocessing
from typing import Any, Dict, List, Optional, Set, Union, Iterable, Tuple

import pandas as pd
from tqdm.auto import tqdm

import math

from cleanvision.dataset.base_dataset import Dataset
from cleanvision.issue_managers import IssueType, register_issue_manager
from cleanvision.issue_managers.image_property import (
    AspectRatioProperty,
    BlurrinessProperty,
    BrightnessProperty,
    ColorSpaceProperty,
    EntropyProperty,
    ImageProperty,
    SizeProperty,
)

# 新增threshold算法跟算f1_score
from cleanvision.utility import (
    f1_for_threshold,
    gamma_mixture_threshold,
    ght_threshold,
    kapur_max_entropy,
    ki_min_error_threshold,
    li_min_cross_entropy,
    mve_otsu,
    otsu_skimage,
)

from cleanvision.utils.base_issue_manager import IssueManager
from cleanvision.utils.constants import (
    IMAGE_PROPERTY,
    IMAGE_PROPERTY_ISSUE_TYPES_LIST,
    MAX_PROCS,
)
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname, update_df


DEFAULT_THRESHOLD_VALUES: Dict[str, float] = {
    IssueType.DARK.value: 0.32,
    IssueType.LIGHT.value: 0.05,
    IssueType.ODD_ASPECT_RATIO.value: 0.35,
    IssueType.LOW_INFORMATION.value: 0.3,
    IssueType.BLURRY.value: 0.29,
}

AUTO_THRESHOLD_ALGORITHMS: List[str] = [
    "li_min_cross_entropy", #通常設為"auto"時會用這個
    "otsu_skimage",
    "ki_min_error",
    "kapur_max_entropy",
    "ght_threshold",
    "mve_otsu",
    "gamma_mixture",
]


def _gamma_mixture_wrapper(values: Iterable[float]) -> Optional[float]:
    threshold, _ = gamma_mixture_threshold(values)
    return threshold


THRESHOLD_ALGORITHMS: Dict[str, Any] = {
    "otsu_skimage": otsu_skimage,
    "otsu": otsu_skimage,
    "ki_min_error": ki_min_error_threshold,
    "ki_min_error_threshold": ki_min_error_threshold,
    "ki": ki_min_error_threshold,
    "li_min_cross_entropy": li_min_cross_entropy,
    "li": li_min_cross_entropy,
    "kapur_max_entropy": kapur_max_entropy,
    "kapur": kapur_max_entropy,
    "ght_threshold": ght_threshold,
    "ght": ght_threshold,
    "mve_otsu": mve_otsu,
    "gamma_mixture": _gamma_mixture_wrapper,
    "gamma_mixture_threshold": _gamma_mixture_wrapper,
    "gamma_gamma": _gamma_mixture_wrapper,
}

def compute_scores(
    index: int,
    dataset: Dataset,
    to_compute: List[str],
    image_properties: Dict[str, ImageProperty],
    params: Dict[str, Dict[str, Any]]
) -> Dict[str, Union[str, int, float]]:
    result: Dict[str, Union[int, str, float]] = {"index": index}
    image = dataset[index]
    if image:
        for issue_type in to_compute:
            param_kwargs = {
                k: v for k, v in params.get(issue_type, {}).items() if "formula" in k
            }
            result = {
                **result,
                **image_properties[issue_type].calculate(image, **param_kwargs)
            }
    return result


def compute_scores_wrapper(args: Dict[str, Any]) -> Dict[str, Union[float, str, int]]:
    return compute_scores(**args)


# Combined all issues which are to be detected using image properties under one class to save time on loading image
@register_issue_manager(IMAGE_PROPERTY)
class ImagePropertyIssueManager(IssueManager):
    issue_name: str = IMAGE_PROPERTY
    visualization: str = "individual_images"

    def __init__(self) -> None:
        super().__init__()
        self.issue_types: List[str] = []
        self._default_thresholds = DEFAULT_THRESHOLD_VALUES.copy()
        self._threshold_ground_truth: Dict[str, Any] = {}
        self._threshold_algorithm_overrides: Dict[str, List[str]] = {}
        self._resolved_thresholds: Dict[str, float] = {}
        self.params = self.get_default_params()
        self.image_properties = self._get_image_properties()

    def get_default_params(self) -> Dict[str, Any]:
        return {
            #IssueType.DARK.value: {"threshold": 0.32},
            #IssueType.LIGHT.value: {"threshold": 0.05},
            #IssueType.ODD_ASPECT_RATIO.value: {"threshold": 0.35},
            IssueType.DARK.value: {
                "threshold": "auto",
                "default_threshold": self._default_thresholds[IssueType.DARK.value],
            },
            IssueType.LIGHT.value: {
                "threshold": "auto",
                "default_threshold": self._default_thresholds[IssueType.LIGHT.value],
            },
            IssueType.ODD_ASPECT_RATIO.value: {
                "threshold": "auto",
                "default_threshold": self._default_thresholds[IssueType.ODD_ASPECT_RATIO.value],
            },
            IssueType.LOW_INFORMATION.value: {
                #"threshold": 0.3,
                "threshold": "auto",
                "default_threshold": self._default_thresholds[IssueType.LOW_INFORMATION.value],
                "normalizing_factor": 0.1,
            },
            IssueType.BLURRY.value: {
                #"threshold": 0.29,
                "threshold": "auto",
                "default_threshold": self._default_thresholds[IssueType.BLURRY.value],
                "normalizing_factor": 0.01,
                "color_threshold": 0.18,
            },
            IssueType.GRAYSCALE.value: {},
            IssueType.ODD_SIZE.value: {"iqr_factor": 3.0},
        }

    def update_params(self, params: Dict[str, Any]) -> None:
        for issue_type in self.params:
            issue_params = params.get(issue_type, {}) or {}

            # 若使用者有提供 default_threshold（且不是 None），就更新內部 _default_thresholds 的數值
            default_threshold = issue_params.get("default_threshold")
            if default_threshold is not None:
                self._default_thresholds[issue_type] = float(default_threshold)

            # 若 key 存在但值是 None，表示使用者明確要求移除預設門檻，從 _default_thresholds 中刪除
            if "default_threshold" in issue_params and default_threshold is None:
                self._default_thresholds.pop(issue_type, None)

            # ground_truth：「每張圖片實際是否有這個 issue」的真實標籤，算f1_score用的，要按圖片index順序排
            ground_truth = issue_params.get("ground_truth")
            if ground_truth is not None or "ground_truth" in issue_params:
                if ground_truth is None:
                    self._threshold_ground_truth.pop(issue_type, None)
                else:
                    self._threshold_ground_truth[issue_type] = ground_truth

            # threshold="auto" 時，用的演算法清單，預設為 AUTO_THRESHOLD_ALGORITHMS
            # 設為"None"或沒給這個key word -> 用預設
            # 有給值 -> 用給的演算法順序，取第一個結果不是None的
            threshold_algorithms = issue_params.get("threshold_algorithms")
            if threshold_algorithms is not None or "threshold_algorithms" in issue_params:
                if threshold_algorithms is None:
                    self._threshold_algorithm_overrides.pop(issue_type, None)
                else:
                    # 使用者給的是單一字串，就轉成 list
                    if isinstance(threshold_algorithms, str):
                        threshold_algorithms = [threshold_algorithms]
                    # 若不是可迭代的型別（list、tuple 等），就報錯
                    if not isinstance(threshold_algorithms, Iterable):
                        raise TypeError(
                            "threshold_algorithms must be a string or an iterable of strings"
                        )
                    # 將演算法名稱全轉成小寫字串存起來 
                    # ex：使用者輸入 "Li", "Otsu" 會變成 ["li", "otsu"]
                    self._threshold_algorithm_overrides[issue_type] = [
                        str(algorithm).lower() for algorithm in threshold_algorithms
                    ]
            
            
            # 過濾值為 None 的鍵、與門檻決策控制相關的鍵
            non_none_params = {
                k: v
                for k, v in issue_params.items()
                if v is not None
                and k
                not in {
                    "ground_truth",
                    "threshold_algorithms",
                    "default_threshold",
                    "resolved_threshold",
                }
            }

            threshold_value = non_none_params.get("threshold")
            if isinstance(threshold_value, str):
                non_none_params["threshold"] = threshold_value.lower()

            self.params[issue_type] = {
                **self.params[issue_type],
                **non_none_params,
            }
            
            if issue_type in self._default_thresholds:
                self.params[issue_type]["default_threshold"] = self._default_thresholds[
                    issue_type
                ]
            else:
                self.params[issue_type].pop("default_threshold", None)

    def _get_image_properties(self) -> Dict[str, ImageProperty]:
        return {
            IssueType.DARK.value: BrightnessProperty(IssueType.DARK.value),
            IssueType.LIGHT.value: BrightnessProperty(IssueType.LIGHT.value),
            IssueType.ODD_ASPECT_RATIO.value: AspectRatioProperty(),
            IssueType.LOW_INFORMATION.value: EntropyProperty(),
            IssueType.BLURRY.value: BlurrinessProperty(),
            IssueType.GRAYSCALE.value: ColorSpaceProperty(),
            IssueType.ODD_SIZE.value: SizeProperty(),
        }

    def _get_defer_set(
        self, issue_types: List[str], agg_computations: pd.DataFrame
    ) -> Set[str]:
        defer_set = set()

        # Add precomputed issues to defer set
        for issue_type in issue_types:
            score_columns = self.image_properties[issue_type].score_columns
            if set(score_columns).issubset(agg_computations.columns):
                defer_set.add(issue_type)

        # Add issues using same property
        if {IssueType.LIGHT.value, IssueType.DARK.value}.issubset(set(issue_types)):
            defer_set.add(IssueType.LIGHT.value)

        return defer_set

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
        self.issues = pd.DataFrame(index=dataset.index)

        self.update_params(params)

        self._resolved_thresholds = {}

        agg_computations = pd.DataFrame(index=dataset.index)
        agg_computations = self._add_prev_computations(agg_computations, imagelab_info)

        defer_set = self._get_defer_set(self.issue_types, agg_computations)

        to_be_computed = list(set(self.issue_types).difference(defer_set))

        new_computations = pd.DataFrame(index=dataset.index)
        if to_be_computed:
            results: List[Dict[str, Union[int, float, str]]] = []
            if n_jobs == 1:
                for idx in tqdm(
                    dataset.index, leave=verbose, desc="Computing scores", smoothing=0
                ):
                    results.append(
                        compute_scores(
                            idx, dataset, to_be_computed, self.image_properties, 
                            params=self.params,
                        )
                    )
            else:
                args = [
                    {
                        "index": idx,
                        "dataset": dataset,
                        "to_compute": to_be_computed,
                        "image_properties": self.image_properties,
                        "params": self.params,
                    }
                    for idx in dataset.index
                ]
                chunksize = max(1, len(args) // MAX_PROCS)
                with multiprocessing.Pool(n_jobs) as p:
                    results = list(
                        tqdm(
                            p.imap_unordered(
                                compute_scores_wrapper, args, chunksize=chunksize
                            ),
                            total=len(dataset),
                            leave=verbose,
                            desc="Computing scores",
                            smoothing=0,
                        )
                    )

                results = sorted(results, key=lambda r: r["index"])

            new_computations = self._aggregate(results)

        agg_computations = update_df(agg_computations, new_computations)

        ordered_issue_types = self._get_dependency_sorted(self.issue_types)

        self.update_issues(agg_computations, ordered_issue_types)
        self.update_info(agg_computations)
        self.update_summary()
        return

    def _get_dependency_sorted(self, issue_types: List[str]) -> List[str]:
        # todo: remove this hacky way, add an ordering
        return sorted(issue_types, reverse=True)

    def update_issues(
        self, agg_computations: pd.DataFrame, issue_types: List[str]
    ) -> None:
        """Updates `self.issues` with score and is_issue columns

        Parameters
        ----------
        agg_computations: pd.DataFrame
            This dataframe contains all computed properties like blurriness, brightness as columns for each image
            that are required for computing issue scores.
        issue_types: List[str]
            List of issue types for which to update `self.issues`
        """
        for issue_type in issue_types:
            score_column_names = self.image_properties[issue_type].score_columns
            score_columns = agg_computations[score_column_names]

            issue_scores = self.image_properties[issue_type].get_scores(
                raw_scores=score_columns,
                issue_type=issue_type,
                #**self.params[issue_type],
                **self._get_score_params(issue_type),
            )

            resolved_threshold = self._resolve_threshold(issue_type, issue_scores)

            is_issue = self.image_properties[issue_type].mark_issue(
                scores=issue_scores,
                issue_type=issue_type,
                #threshold=self.params[issue_type].get("threshold"),
                threshold=resolved_threshold,
            )
            self.issues = self.issues.join(issue_scores)
            self.issues = self.issues.join(is_issue)

    def _add_prev_computations(
        self, agg_computations: pd.DataFrame, info: Dict[str, Any]
    ) -> pd.DataFrame:
        for key in info.keys():
            if key in IMAGE_PROPERTY_ISSUE_TYPES_LIST + ["statistics"]:
                for col in info[key]:
                    if col not in agg_computations:
                        agg_computations = agg_computations.join(info[key][col])
        return agg_computations

    def _aggregate(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        agg_computations = pd.DataFrame(results)
        agg_computations = agg_computations.set_index("index")
        return agg_computations

    def update_info(self, agg_computations: pd.DataFrame) -> None:
        """Updates `self.info` using computed properties

        Parameters
        ----------
        agg_computations: pd.DataFrame
            This dataframe contains all computed properties like blurriness, brightness as columns for each image
            that are required for computing issue scores.
        """
        property_names = {
            issue_type: self.image_properties[issue_type].name
            for issue_type in self.issue_types
        }

        for issue_type in self.issue_types:
            property_name = property_names[issue_type]

            self.info["statistics"][property_name] = agg_computations[
                property_name
            ].describe()

            issue_columns = [
                col for col in agg_computations.columns if col.startswith(property_name)
            ]

            self.info[issue_type] = (
                agg_computations[issue_columns] if len(issue_columns) > 0 else {}
            )

    def update_summary(self) -> None:
        summary_dict = {}
        for issue_type in self.issue_types:
            summary_dict[issue_type] = self._compute_summary(
                self.issues[get_is_issue_colname(issue_type)]
            )
        summary_df = pd.DataFrame.from_dict(summary_dict, orient="index")
        self.summary = summary_df.reset_index()
        self.summary = self.summary.rename(columns={"index": "issue_type"})
        self.summary = self.summary.astype({"num_images": int, "issue_type": str})


    def _get_score_params(self, issue_type: str) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.params[issue_type].items() # 走訪該 issue 的所有參數
            if k not in {"threshold", "default_threshold", "resolved_threshold"} # 排除與門檻決策相關、不要傳進 get_scores 的鍵
        }

    def _resolve_threshold(self, issue_type: str, scores: pd.DataFrame) -> float:
        threshold_setting = self.params[issue_type].get("threshold", "auto") # 取本次門檻設定，預設 "auto"
        
        # 若是數值（固定門檻）轉為 float ，直接回傳固定門檻
        if isinstance(threshold_setting, (int, float)): 
            resolved = float(threshold_setting)
            self.params[issue_type]["resolved_threshold"] = resolved
            self._resolved_thresholds[issue_type] = resolved
            return resolved

        # 若使用者明確設為 None（視為用自動候選清單）
        if threshold_setting is None:
            algorithm_names = self._get_algorithm_candidates(issue_type) # 取演算法清單
            explicit_algorithm = False # 非「顯式指定單一演算法」
        elif isinstance(threshold_setting, str):
            normalized = threshold_setting.lower() # 若是字串，變小寫
            if normalized in {"auto", "best"}: # "auto"/"best" → 用演算法清單機制
                algorithm_names = self._get_algorithm_candidates(issue_type)
                explicit_algorithm = False
            else: # 其他字串 → 視為「顯式指定某一演算法」
                algorithm_names = [normalized]
                explicit_algorithm = True # 只用這一個算
        else:
            raise TypeError(
                f"Threshold for {issue_type} must be a float or string. Received {type(threshold_setting)}."
            )

        values = scores[get_score_colname(issue_type)].astype(float).to_numpy() # 取出此 issue 的分數欄位，轉 float numpy 陣列
        ground_truth = self._get_ground_truth(issue_type, scores.index) # 依當前索引對齊 ground truth（可能為 None）

        candidates: List[Tuple[str, Optional[float], Optional[float]]] = []  # 準備收集候選 (演算法名, 門檻值, F1分數/None)
        # 依序嘗試各演算法
        for name in algorithm_names:
            try: # 計算門檻
                threshold_candidate = self._compute_threshold(name, values)
            except ValueError: # 演算法失敗
                if explicit_algorithm: # 若是顯式指定單一演算法 → 直接報錯
                    raise
                continue
            if threshold_candidate is None: # 演算法回傳 None → 視為失敗
                continue
            candidate_score = ( # 若有標記 → 計算 F1；否則記 None
                f1_for_threshold(threshold_candidate, values, ground_truth)
                if ground_truth is not None
                else None
            )
            candidates.append((name, threshold_candidate, candidate_score)) #結果存起來

        default_threshold = self.params[issue_type].get("default_threshold")  # 讀入使用者給的 default_threshold
        if default_threshold is None:
            default_threshold = self._default_thresholds.get(issue_type) # 若沒有，用預設
        if default_threshold is not None:
            default_candidate_score = ( # 有ground_truth就對 default 也算 F1
                f1_for_threshold(default_threshold, values, ground_truth)
                if ground_truth is not None
                else None
            )
            candidates.append(("default", float(default_threshold), default_candidate_score)) # 把 default 也當成一個候選

        if not candidates: # 若完全沒有候選（全失敗）
            raise ValueError(
                f"Unable to determine a threshold for issue type '{issue_type}'."
            )

        # 若有 ground truth → 挑 F1 最好
        best_candidate = None
        best_score = float("-inf")
        for idx, candidate in enumerate(candidates):
            algo_name, threshold_value, score_value = candidate
            # 沒有 F1（無標記）→ 略過此段比較
            if score_value is None: 
                continue
            if score_value > best_score or (score_value == best_score and best_candidate and idx < best_candidate[3]):
                best_candidate = (algo_name, threshold_value, score_value, idx) # 同分時選順序較前面的
                best_score = score_value

        if best_candidate is not None: #有最佳F1 -> 用他
            algo_name, resolved_threshold, _, _ = best_candidate
        else: # 沒有 F1（無標記）→ 用候選清單順序挑第一個可用
            resolved_threshold = self._choose_first_available_threshold(
                algorithm_names, candidates
            )
            # 從候選中找出是哪個演算法被採用
            algo_name = next(
                (name for name, val, _ in candidates if val is not None and math.isclose(val, resolved_threshold, rel_tol=1e-9, abs_tol=1e-12)),
                "unknown"
            )

        assert resolved_threshold is not None
        resolved_threshold = float(resolved_threshold)
        self.params[issue_type]["resolved_threshold"] = resolved_threshold # 記錄實際採用的門檻（方便之後檢視/輸出）
        self._resolved_thresholds[issue_type] = resolved_threshold
        
        # 印出實際使用的演算法
        print(f"[{issue_type}] resolved threshold = {resolved_threshold:.4f} (algorithm: {algo_name})")

        return resolved_threshold

    def _get_algorithm_candidates(self, issue_type: str) -> List[str]:
        override = self._threshold_algorithm_overrides.get(issue_type) # 看此 issue 是否有自訂演算法清單
        if override:
            return [algorithm.lower() for algorithm in override] # 有覆寫 → 回傳小寫化的清單
        return list(AUTO_THRESHOLD_ALGORITHMS) #不然用預設清單

    def _compute_threshold(
        self, algorithm_name: str, values: Iterable[float]
    ) -> Optional[float]:
        normalized = algorithm_name.lower() # 名稱小寫化
        if normalized not in THRESHOLD_ALGORITHMS: # 出現沒有的演算法名，報錯
            raise ValueError(f"Unknown threshold algorithm '{algorithm_name}'.")
        algorithm = THRESHOLD_ALGORITHMS[normalized] # 取得演算法函式
        try:
            result = algorithm(values)  # 執行演算法，回傳門檻或 (門檻, 其他資訊)
        except Exception as exc:  # pragma: no cover - unexpected algorithm failure
            raise ValueError(
                f"Threshold algorithm '{algorithm_name}' failed with error: {exc}"
            )

        if isinstance(result, tuple): # 若回傳 tuple，取第 1 個元素為門檻
            threshold_value = result[0]
        else:
            threshold_value = result # 否則直接視為門檻
        if threshold_value is None: # None 視為失敗
            return None
        try:
            return float(threshold_value) # 嘗試轉成 float
        except (TypeError, ValueError) as exc: # 若回傳非數值，報錯
            raise ValueError(
                f"Threshold algorithm '{algorithm_name}' returned a non-numeric value"
            ) from exc

    def _choose_first_available_threshold(
        self,
        algorithm_names: Iterable[str],
        candidates: List[Tuple[str, Optional[float], Optional[float]]],
    ) -> float:
        candidate_lookup = {name: value for name, value, _ in candidates} # 建立候選查表：演算法名 → 門檻值
        for algorithm in algorithm_names: # 依「候選清單順序」找第一個有值的門檻
            if algorithm in candidate_lookup and candidate_lookup[algorithm] is not None:
                return float(candidate_lookup[algorithm])
        return float(candidates[-1][1]) # 保底用最後一個候選(理論上不會執行到)

    def _get_ground_truth(
        self, issue_type: str, index: pd.Index
    ) -> Optional[pd.Series]:
        if issue_type not in self._threshold_ground_truth:
            return None
        labels = self._threshold_ground_truth[issue_type] # 取出標記（可能是 Series 或 list）
        if labels is None: # 設為 None → 視為移除標記
            return None

        if isinstance(labels, pd.Series): #是Series
            if len(labels) != len(index): # Series長度不符，報錯
                raise ValueError(
                    f"Ground truth for '{issue_type}' must have the same length as the scores."
                )
            if labels.index.equals(index): # 索引完全一致 → 直接對齊
                aligned = labels.loc[index]
            else:
                try:
                    aligned = labels.reindex(index) # 嘗試 reindex 對齊 scores 的索引
                except Exception as exc:
                    raise ValueError(
                        f"Unable to align ground truth for '{issue_type}' with score index."
                    ) from exc
                if aligned.isna().any(): # reindex 後若出現缺值，報錯
                    raise ValueError(
                        f"Ground truth for '{issue_type}' contains missing values after alignment."
                    )
        else: # 若是 list / array-like
            aligned_series = pd.Series(labels)
            if len(aligned_series) != len(index):
                raise ValueError(
                    f"Ground truth for '{issue_type}' must have the same length as the scores."
                )
            aligned_series.index = index
            aligned = aligned_series

        if aligned.isna().any():
            raise ValueError(
                f"Ground truth for '{issue_type}' contains missing values."
            )

        unique_values = set(aligned.unique())
        if not unique_values.issubset({0, 1, False, True}):
            raise ValueError(
                f"Ground truth for '{issue_type}' must be boolean or contain 0/1 values."
            )

        return aligned.astype(int) # 轉成 0/1 的整數標記，回傳