import argparse
from io import BytesIO

import onnx
import torch
from typing import Tuple

import torch
import torch.nn as nn
from torch import Graph, Tensor, Value
from ultralytics import YOLO

try:
    import onnxsim
except ImportError:
    onnxsim = None

class YOLOv8Pose(nn.Module):
    export = True
    shape = None
    dynamic = False

    def __init__(self, weights, device):
        super().__init__()
        self.device = device
        self.model = YOLO(weights).to(self.device).model.fuse().eval()
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        out, _ = self.model(x)
        boxes, scores, kps = out.split((4,1,51), 1)
        boxes = (boxes.transpose(1,2) @ self.convert_matrix)
        return boxes, scores.transpose(1,2), kps.transpose(1,2)
    
    def export(self, save_path, opset_version=11, sim=True):
        fake_input = torch.randn(1,3,640,640).to(self.device)
        for _ in range(2):
            self.forward(fake_input)
        with BytesIO() as f:
            torch.onnx.export(
                self,
                fake_input,
                f,
                opset_version=opset_version,
                input_names=['images'],
                output_names=['bboxes', 'scores', 'kps'])
            f.seek(0)
            onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        if sim:
            try:
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')
        onnx.save(onnx_model, save_path)
        print(f'ONNX export success, saved as {save_path}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch weights')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    model = YOLOv8Pose(args.weights, 'cpu')
    save_path = args.weights.replace('.pt', '.onnx')
    model.export(save_path, args.opset, args.sim)