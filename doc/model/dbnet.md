# DBNet

## Get pytorch model
The pytorch implementation is [DBNet.pytorch](https://github.com/BaofengZan/DBNet.pytorch).

## Export
### ONNX
```
git clone https://github.com/Huntersdeng/CXX-DeepLearning-Inference.git
git clone https://github.com/BaofengZan/DBNet.pytorch.git
cp CXX-DeepLearning-Inference/model/ocr/scripts/dbnet_export.py DBNet.pytorch
cd DBNet.pytorch
python3 dbnet_export.py --sim --weights=path-to-weights
```
It's an onnx model with dynamic axes, you can check the onnx model in [netron](netron.app).
- inputs
    - images (float32[1,3,height,width])
- outputs
    - output (float32[Resizeoutput_dim_0,Resizeoutput_dim_1,Resizeoutput_dim_2,Resizeoutput_dim_3])

### TensorRT
```
${tensorrt-install-path}/bin/trtexec \                                                                           
--onnx=DBNet.onnx \
--saveEngine=DBNet.engine \
--fp16 \
--minShapes=images:1x3x608x608 \
--maxShapes=images:1x3x1440x1440 \
--optShapes=images:1x3x640x1152
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
config/ocr/det/dbnet.yaml
```
model_name: "dbnet"
model_path: "../weights/ocr/det/DBNet.onnx"
framework: "ONNX"
box_thres: 0.5
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
config/ocr/det/dbnet.yaml
```
model_name: "dbnet"
model_path: "../weights/ocr/det/DBNet.engine"
framework: "TensorRT"
box_thres: 0.5
```
#### Run
```
cd build
./test/ocr_test
```