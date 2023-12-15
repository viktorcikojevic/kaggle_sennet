from typing import List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass


def split_to_chunks(items, chunk_size: int):
    ans = []
    for i in range(0, len(items), chunk_size):
        ans.append(items[i: i+chunk_size])
    return ans


def split_to_n_chunks(items, n_chunks: int):
    return split_to_chunks(items, int(np.ceil(len(items) / n_chunks)))


def split_crop_into_chunks(
        channel_start: int,
        channel_end: int,
        chunk_boundaries: List[Tuple[int, int]],
) -> List[Tuple[int, int, int, int]]:
    ret = []
    for start, end in chunk_boundaries:
        if start > channel_end or end <= channel_start:
            ret.append(None)
            continue
        intersected_start = max(channel_start, start)
        intersected_end = min(channel_end, end)
        crop_start = intersected_start - channel_start
        crop_end = intersected_end - channel_start
        fill_start = intersected_start - start
        fill_end = intersected_end - start
        ret.append((
            crop_start,
            crop_end,
            fill_start,
            fill_end,
        ))
    return ret


@dataclass
class TensorChunk:
    tensor: torch.Tensor
    crop_start: int
    crop_end: int
    fill_start: int
    fill_end: int


class TensorDistributor:
    def __init__(self, n_channels: int, n_chunks: int):
        self.n_channels = n_channels
        self.n_chunks = n_chunks
        self.chunk_boundaries = [
            (chunk[0], chunk[-1] + 1)  # note: this can be optimised a lot, but boy I don't really care :P
            for chunk in split_to_n_chunks(list(range(self.n_channels)), self.n_chunks)
        ]

    def distribute_tensor(self, tensor: torch.Tensor, channel_start: int, channel_end: int) -> List[Optional[TensorChunk]]:
        """
        :param tensor: (c, h, w)
        :param channel_start: int
        :param channel_end: int
        :return: List[torch.Tensor]
        """
        chunk_intersections = split_crop_into_chunks(int(channel_start), int(channel_end), self.chunk_boundaries)
        tensor_chunks = []
        for c in chunk_intersections:
            if c is None:
                tensor_chunks.append(None)
                continue

            crop_start, crop_end, fill_start, fill_end = c
            chunk = TensorChunk(
                tensor=tensor[crop_start: crop_end, :, :],
                crop_start=crop_start,
                crop_end=crop_end,
                fill_start=fill_start,
                fill_end=fill_end,
            )
            tensor_chunks.append(chunk)
        return tensor_chunks


if __name__ == "__main__":
    _n_chunks = 5
    _n_channels = 100
    _channel_start = 23
    _channel_end = 94

    _chunk_boundaries = [
        (chunk[0], chunk[-1] + 1)
        for chunk in split_to_n_chunks(list(range(_n_channels)), _n_chunks)
    ]
    _res = split_crop_into_chunks(
        channel_start=_channel_start,
        channel_end=_channel_end,
        chunk_boundaries=_chunk_boundaries,
    )
    print(_res)
