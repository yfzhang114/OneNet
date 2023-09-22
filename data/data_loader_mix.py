# coding=utf-8
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import ImageFolder
import bisect
import torch
from typing import TypeVar

T_co = TypeVar('T_co', covariant=True)
class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets) -> None:
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.domain_label = torch.randint(low=0, high=len(datasets), size=(self.cummulative_sizes[-1],))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.datasets[dataset_idx].indices[sample_idx]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes