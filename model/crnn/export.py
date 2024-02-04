import onnx
import torch
import argparse
from io import BytesIO
import torch.nn as nn

import models.crnn as crnn
try:
    import onnxsim
except ImportError:
    onnxsim = None

class CRNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.crnn = crnn.CRNN(crnn.CRNN(32, 1, 37, 256))

    def forward(self, x):
        output = self.crnn(x)
        scores, labels = output.transpose(0,1).max(dim=-1, keepdim=True)
        return labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch crnn weights')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[1, 1, 32, 100],
                        help='Model input shape only for api builder')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Export ONNX device')
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args

def main(args):
    model_path = args.weights

    model = CRNN()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)

    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=['images'],
            output_names=['output'])
        f.seek(0)
        onnx_model = onnx.load(f)

    onnx.checker.check_model(onnx_model)
    save_path = args.weights.replace('.pth', '.onnx')

    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')

if __name__ == '__main__':
    main(parse_args())