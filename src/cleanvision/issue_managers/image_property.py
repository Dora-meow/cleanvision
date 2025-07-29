import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, overload

import numpy as np
import pandas as pd
from PIL import ImageFilter, ImageStat
from PIL.Image import Image

from cleanvision.issue_managers import IssueType
from cleanvision.utils.constants import MAX_RESOLUTION_FOR_BLURRY_DETECTION
from cleanvision.utils.utils import get_is_issue_colname, get_score_colname


class ImageProperty(ABC):
    name: str

    @property
    @abstractmethod
    def score_columns(self) -> List[str]:
        pass

    @staticmethod
    def check_params(**kwargs: Any) -> None:
        allowed_kwargs: Dict[str, Any] = {
            "image": Image,
            "dark_issue_data": pd.DataFrame,
            "threshold": float,
        }

        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise ValueError(f"{name} is not a valid keyword argument.")
            if value is not None and not isinstance(value, allowed_kwargs[name]):
                raise ValueError(
                    f"Valid type for keyword argument {name} can only be {allowed_kwargs[name]}. {name} cannot be type {type(name)}. "
                )

    @abstractmethod
    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        raise NotImplementedError

    @abstractmethod
    def get_scores(
        self, raw_scores: pd.DataFrame, issue_type: str, **kwargs: Any
    ) -> Any:
        self.check_params(**kwargs)
        return

    def mark_issue(
        self,
        scores: pd.DataFrame,
        issue_type: str,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        is_issue = pd.DataFrame(index=scores.index)
        is_issue_colname, score_colname = get_is_issue_colname(
            issue_type
        ), get_score_colname(issue_type)
        is_issue[is_issue_colname] = scores[score_colname] < threshold
        return is_issue


def calc_avg_brightness(image: Image, brightness_formula: int = 1) -> float:
    stat = ImageStat.Stat(image)
    try:
        red, green, blue = stat.mean
    except ValueError:
        red, green, blue = (
            stat.mean[0],
            stat.mean[0],
            stat.mean[0],
        )  # deals with black and white images

    cur_bright: float = calculate_brightness(red, green, blue, brightness_formula=brightness_formula)
    return cur_bright


@overload
def calculate_brightness(red: float, green: float, blue: float, brightness_formula: int = 1) -> float: ...


@overload
def calculate_brightness(
    red: "np.ndarray[Any, Any]",
    green: "np.ndarray[Any, Any]",
    blue: "np.ndarray[Any, Any]",
    brightness_formula: int = 1
) -> "np.ndarray[Any, Any]": ...


def calculate_brightness(
    red: Union[float, "np.ndarray[Any, Any]"],
    green: Union[float, "np.ndarray[Any, Any]"],
    blue: Union[float, "np.ndarray[Any, Any]"],
    brightness_formula: int = 1
) -> Union[float, "np.ndarray[Any, Any]"]:
    #print('current brightness_formula:', brightness_formula, '\n')
    if brightness_formula == 1: #套件原本的公式
      #print("brightness_formula 1 \n")
      cur_bright = (
          np.sqrt(0.241 * (red * red) + 0.691 * (green * green) + 0.068 * (blue * blue))
      ) / 255
    elif brightness_formula == 2: #常用公式
      #print("brightness_formula 2\n")
      cur_bright = (
          0.2126 * red + 0.7152 * green + 0.0722 * blue
      ) / 255
    elif brightness_formula == 3: #另一個常用公式
      cur_bright = (
          0.299 * red + 0.587 * green + 0.114 * blue
      ) / 255
    elif brightness_formula == 4: #套件公式改係數
      cur_bright = (
          np.sqrt(0.299 * (red * red) + 0.587 * (green * green) + 0.114 * (blue * blue))
      ) / 255
    elif brightness_formula == 5: #取平均
      cur_bright = (
          (red + green + blue) / 3
      ) / 255
    else:
        raise ValueError(f"Unknown formula: {brightness_formula}")

    return cur_bright


def calc_percentile_brightness(
    image: Image, percentiles: List[int], brightness_formula: int = 1
) -> "np.ndarray[Any, Any]":
    imarr = np.asarray(image)
    if len(imarr.shape) == 3:
        r, g, b = (
            imarr[:, :, 0].astype("int"),
            imarr[:, :, 1].astype("int"),
            imarr[:, :, 2].astype("int"),
        )
        pixel_brightness = calculate_brightness(
            r, g, b, brightness_formula=brightness_formula
        )  #原本的公式(預設): np.sqrt(0.241 * r * r + 0.691 * g * g + 0.068 * b * b)
    else:
        pixel_brightness = imarr / 255.0
    perc_values: "np.ndarray[Any, Any]" = np.percentile(pixel_brightness, percentiles)
    return perc_values


class BrightnessProperty(ImageProperty):
    name: str = "brightness"

    @staticmethod
    def check_params(**kwargs: Any) -> None:
        allowed_kwargs: Dict[str, Any] = {
            "image": Image,
            "dark_issue_data": pd.DataFrame,
            "threshold": float,
            "brightness_formula": int,
            "brightness_percentile": int,
            "use_avg": bool,
        }

        for name, value in kwargs.items():
            if name not in allowed_kwargs:
                raise ValueError(f"{name} is not a valid keyword argument.")
            if value is not None and not isinstance(value, allowed_kwargs[name]):
                raise ValueError(
                    f"Valid type for keyword argument {name} can only be {allowed_kwargs[name]}. {name} cannot be type {type(name)}. "
                )

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self, issue_type: str) -> None:
        self.issue_type = issue_type
        self._score_columns = [f"brightness_perc_{p}" for p in [1, 5, 10, 15, 25, 30, 40, 50, 60, 75, 90, 95, 99]]
        self._score_columns.append(self.name)

    def calculate(self, image: Image, brightness_formula: int = 1) -> Dict[str, Union[float, str]]:
        #print('hello')
        percentiles = [1, 5, 10, 15, 25, 30, 40, 50, 60, 75, 90, 95, 99]
        perc_values = calc_percentile_brightness(image, percentiles=percentiles, brightness_formula=brightness_formula)
        raw_values = {
            f"brightness_perc_{p}": value for p, value in zip(percentiles, perc_values)
        }
        raw_values[self.name] = calc_avg_brightness(image, brightness_formula=brightness_formula)
        #print(raw_values)
        return raw_values

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)

        use_avg = kwargs.get("use_avg")
        #只有use_avg是True的時候才會執行
        if use_avg:  
            self._score_columns = ['brightness']
            print('self._score_columns: ',self._score_columns)
            scores = pd.DataFrame(index=raw_scores.index)
            scores[get_score_colname(issue_type)] = raw_scores['brightness']
            return scores

        #其他情況(use_avg是None或False)
        brightness_percentile = kwargs.get("brightness_percentile")
        if brightness_percentile is None:
            brightness_percentile = 99 if issue_type == IssueType.DARK.value else 5
        column_name = f"brightness_perc_{brightness_percentile}"
        self._score_columns = [column_name]
        print('self._score_columns: ',self._score_columns)
        
        scores = pd.DataFrame(index=raw_scores.index)
        #print('raw_scores: ', raw_scores)
        #print('scores: ',scores)
        
        if issue_type == IssueType.DARK.value:
            scores[get_score_colname(issue_type)] = raw_scores[column_name]
        else:
            scores[get_score_colname(issue_type)] = 1 - raw_scores[column_name]
        
        #print('scores(2): ',scores)
        return scores


def calc_aspect_ratio(image: Image) -> float:
    width, height = image.size
    size_score = min(width / height, height / width)  # consider extreme shapes
    assert isinstance(size_score, float)
    return size_score


class AspectRatioProperty(ImageProperty):
    name: str = "aspect_ratio"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_aspect_ratio(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = raw_scores[self.score_columns[0]]
        return scores


def calc_entropy(image: Image) -> float:
    entropy = image.entropy()
    assert isinstance(
        entropy, float
    )  # PIL does not have type ann stub so need to assert function return
    return entropy


class EntropyProperty(ImageProperty):
    name: str = "entropy"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_entropy(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        normalizing_factor: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert raw_scores is not None
        scores = pd.DataFrame(index=raw_scores.index)
        scores_data = normalizing_factor * raw_scores[self.score_columns[0]]
        scores_data[scores_data > 1] = 1
        scores[get_score_colname(issue_type)] = scores_data
        return scores


def calc_blurriness(gray_image: Image) -> float:
    edges = get_edges(gray_image)
    blurriness = ImageStat.Stat(edges).var[0]
    return np.sqrt(blurriness)  # type:ignore


def calc_std_grayscale(gray_image: Image) -> float:
    return np.std(gray_image.histogram())  # type: ignore


class BlurrinessProperty(ImageProperty):
    name = "blurriness"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name, "blurriness_grayscale_std"]
        self.max_resolution = MAX_RESOLUTION_FOR_BLURRY_DETECTION

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        ratio = max(image.width, image.height) / self.max_resolution
        if ratio > 1:
            resized_image = image.resize(
                (max(int(image.width // ratio), 1), max(int(image.height // ratio), 1))
            )
        else:
            resized_image = image.copy()
        gray_image = resized_image.convert("L")
        return {
            self.name: calc_blurriness(gray_image),
            "blurriness_grayscale_std": calc_std_grayscale(gray_image),
        }

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        normalizing_factor: float = 1.0,
        color_threshold: float = 1.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        blur_scores = 1 - np.exp(-1 * raw_scores[self.name] * normalizing_factor)
        std_scores = 1 - np.exp(
            -1 * raw_scores["blurriness_grayscale_std"] * normalizing_factor
        )
        std_scores[std_scores <= color_threshold] = 0

        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = np.minimum(blur_scores + std_scores, 1)
        return scores


def get_edges(gray_image: Image) -> Image:
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    return edges


def calc_color_space(image: Image) -> str:
    return get_image_mode(image)


def calc_image_area_sqrt(image: Image) -> float:
    w, h = image.size
    return math.sqrt(w) * math.sqrt(h)


class ColorSpaceProperty(ImageProperty):
    name = "color_space"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_color_space(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert raw_scores is not None
        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = [
            0 if x == "L" else 1 for x in raw_scores[self.score_columns[0]]
        ]
        return scores

    def mark_issue(
        self, scores: pd.DataFrame, issue_type: str, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        is_issue = pd.DataFrame(index=scores.index)
        is_issue_colname, score_colname = get_is_issue_colname(
            issue_type
        ), get_score_colname(issue_type)

        is_issue[is_issue_colname] = (1 - scores[score_colname]).astype("bool")
        return is_issue


class SizeProperty(ImageProperty):
    name = "size"

    @property
    def score_columns(self) -> List[str]:
        return self._score_columns

    def __init__(self) -> None:
        self._score_columns = [self.name]
        self.threshold = 0.5  # todo: this ensures that the scores are evenly distributed across the range

    def calculate(self, image: Image) -> Dict[str, Union[float, str]]:
        return {self.name: calc_image_area_sqrt(image)}

    def get_scores(
        self,
        raw_scores: pd.DataFrame,
        issue_type: str,
        iqr_factor: float = 3.0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        super().get_scores(raw_scores, issue_type, **kwargs)
        assert raw_scores is not None

        size = raw_scores[self.name]
        q1, q3 = np.percentile(size, [25, 75])
        size_iqr = q3 - q1
        min_threshold, max_threshold = (
            q1 - iqr_factor * size_iqr,
            q3 + iqr_factor * size_iqr,
        )
        mid_threshold = (min_threshold + max_threshold) / 2
        threshold_gap = max_threshold - min_threshold
        distance = np.absolute(size - mid_threshold)

        if threshold_gap > 0:
            norm_value = threshold_gap
            self.threshold = 0.5
        elif threshold_gap == 0:
            norm_value = mid_threshold
            self.threshold = 1.0
        else:
            raise ValueError("threshold_gap should be non negative")

        norm_dist = distance / norm_value
        score_values = 1 - np.clip(norm_dist, 0, 1)

        scores = pd.DataFrame(index=raw_scores.index)
        scores[get_score_colname(issue_type)] = score_values
        return scores

    def mark_issue(
        self, scores: pd.DataFrame, issue_type: str, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        threshold = self.threshold if threshold is None else threshold
        is_issue_colname, score_colname = get_is_issue_colname(
            issue_type
        ), get_score_colname(issue_type)

        is_issue = pd.DataFrame(index=scores.index)
        is_issue[is_issue_colname] = scores[score_colname] < threshold
        return is_issue


def get_image_mode(image: Image) -> str:
    if image.mode:
        image_mode = image.mode
        assert isinstance(image_mode, str)
        if image_mode != "L":
            imarr = np.asarray(image)
            #mode是RGB也要檢查是否三通道相同
            if len(imarr.shape) == 3 and (np.diff(imarr.reshape(-1, 3).T, axis=0) == 0).all() :
                image_mode = "L"
        return image_mode
    else:
        imarr = np.asarray(image)
        if len(imarr.shape) == 2 or (
            len(imarr.shape) == 3
            and (np.diff(imarr.reshape(-1, 3).T, axis=0) == 0).all()
        ):
            return "L"
        else:
            return "UNK"

