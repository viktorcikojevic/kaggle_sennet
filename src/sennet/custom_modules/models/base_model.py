from dataclasses import dataclass
from abc import ABC
import torch


@dataclass
class SegmentorOutput:
    pred: torch.Tensor
    take_indices_start: int
    take_indices_end: int


class Base3DSegmentor(ABC, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def get_name(self) -> str:
        """

        :return: str, name of the model to be logged on wandb
        """
        pass

    def predict(self, img: torch.Tensor) -> SegmentorOutput:
        """

        :param img: torch.Tensor: (b, c, z, h, w)
        :return: SegmentorOutput:
            - pred: (b, z1, h, w)
            - take_indices: (z, ): which z channels from the input this is meant to predict, starting from 0
        """
        pass
