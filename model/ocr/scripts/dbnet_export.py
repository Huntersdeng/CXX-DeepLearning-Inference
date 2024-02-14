import torch
import onnx
from io import BytesIO

try:
    import onnxsim
except ImportError:
    onnxsim = None

from models import build_model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch dbnet weights')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Export ONNX device')
    args = parser.parse_args()
    return args

def main(args):
    checkpoint = torch.load(args.weights, map_location=args.device)
    config = checkpoint['config']
    config['arch']['backbone']['pretrained'] = False
    model = build_model(config['arch'])
    model.to(args.device)

    fake_input = torch.randn((1, 3, 640, 640)).to(args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace('.pth', '.onnx') 

    with BytesIO() as f:
        torch.onnx.export(model, fake_input, f, verbose=False, opset_version=12, input_names=['images'],
                        output_names=['output'], 
                        dynamic_axes={"images": {2: "height", 3: "width"}})
        f.seek(0)
        onnx_model = onnx.load(f)

    onnx.checker.check_model(onnx_model)  # check onnx model
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print('ONNX export success, saved as %s' % save_path)
    


if __name__ == '__main__':
    main(parse_args())
