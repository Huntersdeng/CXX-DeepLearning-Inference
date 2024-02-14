# ABINet

## Get pytorch model
The pytorch implementation is [ABINet](https://github.com/FangShancheng/ABINet).

## Export
### ONNX
```
git clone https://github.com/Huntersdeng/CXX-DeepLearning-Inference.git
git clone https://github.com/FangShancheng/ABINet.git
cp CXX-DeepLearning-Inference/model/ocr/scripts/abinet_export.py ABINet
cd ABINet
python3 abinet_export.py --sim --weights=path-to-weights
```
You can checkout the onnx model in [netron](netron.app).
- inputs
    - images (float32[1,3,32,128])
- outputs
    - output (float32[1,26,1])

### TensorRT
```
${tensorrt-install-path}/bin/trtexec                                                             
--onnx=path-to-your-onnx-model \
--saveEngine=save-path \
--fp16
```

## Inference
### ONNXRuntime
#### Build
```
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_TENSORRT=OFF
make
```
#### Config
config/ocr/rec/attn.yaml for abinet model
```
model_name: "abinet"
model_path: "../weights/ocr/best-train-abinet.onnx"
framework: "ONNX"
input_size: [128,32]  # (width, height)
input_channel: 3
alphabet: "abcdefghijklmnopqrstuvwxyz0123456789"
output_size: 26
```
#### Run
```
cd build
./test/ocr_test
```

### TensorRT
```
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_TENSORRT=ON
make
```
#### Config
config/ocr/rec/attn.yaml for abinet model
```
model_name: "abinet"
model_path: "../weights/ocr/best-train-abinet.engine"
framework: "TensorRT"
input_size: [128,32]  # (width, height)
input_channel: 3
alphabet: "abcdefghijklmnopqrstuvwxyz0123456789"
output_size: 26
```
#### Run
```
cd build
./test/ocr_test
```