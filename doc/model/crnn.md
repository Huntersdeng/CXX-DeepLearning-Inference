# CRNN

## Get pytorch model
The pytorch implementation is [crnn.pytorch](https://github.com/meijieru/crnn.pytorch).

## Export
### ONNX
```
git clone https://github.com/Huntersdeng/CXX-DeepLearning-Inference.git
git clone https://github.com/meijieru/crnn.pytorch.git
cp CXX-DeepLearning-Inference/model/ocr/scripts/crnn_export.py crnn.pytorch
cd crnn.pytorch
python3 crnn_export.py --weights=crnn.pth --sim
```
You can checkout the onnx model in [netron](netron.app).
- inputs
    - images (float32[1,1,32,100])
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
config/ocr/ctc.yaml for crnn model
```
model_name: "crnn"
model_path: "../weights/ocr/crnn.onnx"
framework: "ONNX"
input_size: [100,32]  # (width, height)
input_channel: 1
alphabet: "0123456789abcdefghijklmnopqrstuvwxyz"
output_size: 26
```
#### Run
```
cd build
./test/ocr_test
```
You can see output like:
```
Input: 
images: 3200
Output: 
output: 26
../test/image/ocr/demo.png: available
cost 7.9420 ms
Destruct ocr model
```

### TensorRT
```
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_TENSORRT=ON
make
```
#### Config
config/ocr/ctc.yaml for crnn model
```
model_name: "crnn"
model_path: "../weights/ocr/crnn.engine"
framework: "TensorRT"
input_size: [100,32]  # (width, height)
input_channel: 1
alphabet: "0123456789abcdefghijklmnopqrstuvwxyz"
output_size: 26
```
#### Run
```
cd build
./test/ocr_test
```
You can see output like:
```
Input bind name: images
Output bind name: output
model warmup 10 times
../test/image/ocr/demo.png: available
cost 16.3920 ms
Destruct ocr model
```