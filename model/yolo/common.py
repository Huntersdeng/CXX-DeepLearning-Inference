# copyed from https://github.com/triple-Mu/YOLOv8-TensorRT

from typing import Tuple

import torch
import torch.nn as nn
from torch import Graph, Tensor, Value

def make_anchors(feats: Tensor,
                 strides: Tensor,
                 grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device,
                          dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device,
                          dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class C2f(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))