import os
import onnx
import torch
import argparse
from io import BytesIO
import torch.nn as nn

from utils import Config

try:
    import onnxsim
except ImportError:
    onnxsim = None

class ONNXModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.get_model(config, device)
        self.load(config.model_checkpoint, device=device)
        print('loading pretrained model from %s' % config.model_checkpoint)

    def forward(self, x):
        logits, length = self.model(x)
        scores, labels = logits.max(dim=-1, keepdim=True)
        return labels
    
    def get_model(self, config, device):
        import importlib
        names = config.model_name.split('.')
        module_name, class_name = '.'.join(names[:-1]), names[-1]
        cls = getattr(importlib.import_module(module_name), class_name)
        self.model = cls(config).eval().to(device)

    def load(self, file, device=None, strict=True):
        if device is None:
            device = 'cpu'
        elif isinstance(device, int):
            device = torch.device('cuda', device)
        assert os.path.isfile(file)
        state = torch.load(file, map_location=device)
        if set(state.keys()) == {'model', 'opt'}:
            state = state['model']
        self.model.load_state_dict(state, strict=strict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        default='workdir/train-abinet/best-train-abinet.pth',
                        help='PyTorch weights')
    parser.add_argument('--opset',
                        type=int,
                        default=13,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[1, 3, 32, 128],
                        help='Model input shape only for api builder')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                        help='path to config file')
    parser.add_argument('--model_eval', type=str, default='alignment',
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args

def main(args):
    config = Config(args.config)
    if args.weights is not None: config.model_checkpoint = args.weights
    if args.model_eval is not None: config.model_eval = args.model_eval
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    device = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'
    config.export = True

    model = ONNXModel(config, device)
    fake_input = torch.randn(args.input_shape).to(device)
    for _ in range(2):
        model(fake_input)

    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
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