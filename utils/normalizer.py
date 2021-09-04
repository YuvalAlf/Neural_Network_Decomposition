from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence, List, Dict

import pandas as pd
from pandas import DataFrame

from utils.functional_utils import apply_to_result


@dataclass
class Normalizer:
    center: float
    std: float

    @staticmethod
    def get_normalizer(data: Sequence[float]) -> Normalizer:
        center = mean(data)
        std = stdev(data)
        return Normalizer(center, std)

    @apply_to_result(list)
    def normalize(self, data: Sequence[float]) -> List[float]:
        for value in data:
            yield (value - self.center) / self.std

    @apply_to_result(list)
    def denormalize(self, data: Sequence[float]) -> List[float]:
        for value in data:
            yield (value + self.center) * self.std


@dataclass
class DataFrameNormalizer:
    normalizers: Dict[str, Normalizer]

    @staticmethod
    def normalize_data_frame(data_frame: DataFrame) -> (DataFrame, Normalizer):
        normalizers = DataFrameNormalizer.calc_data_frame_normalizers(data_frame)
        data_frame_normalizer = DataFrameNormalizer(normalizers)
        return data_frame_normalizer.normalize(data_frame), data_frame_normalizer

    def normalize(self, data_frame: DataFrame) -> DataFrame:
        normalized_data_frame = pd.DataFrame()
        for column, normalizer in self.normalizers.items():
            normalized_data_frame[column] = normalizer.normalize(data_frame[column])
        return normalized_data_frame

    def denormalize(self, data_frame: DataFrame) -> DataFrame:
        normalized_data_frame = pd.DataFrame()
        for column, normalizer in self.normalizers.items():
            normalized_data_frame[column] = normalizer.denormalize(data_frame[column])
        return normalized_data_frame

    @staticmethod
    @apply_to_result(dict)
    def calc_data_frame_normalizers(data_frame: DataFrame) -> Dict[str, Normalizer]:
        for column in data_frame.columns:
            normalizer = Normalizer.get_normalizer(data_frame[column])
            yield column, normalizer
