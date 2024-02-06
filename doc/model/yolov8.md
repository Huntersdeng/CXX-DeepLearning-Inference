# YOLOv8

## Prepare
```
python3 -m pip install ultralytics, onnx, onnxsim
```

## Get pytorch model
```
# detect
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
# seg
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt
```

## Export
### ONNX
#### Detect
```
# without-nms
python3 model/yolov8/export-det.py --weights=path-to-your-weights --sim
# with trt-nms plugin (only for tensorrt transfer, not support to inference with onnxruntime C++)
python3 model/yolov8/export-det.py --weights=path-to-your-weights --sim --trt-nms
```

#### Segment
```
python3 export-seg.py --weights=path-to-your-weights --sim
```

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
config/yolo/yolov8_normal.yaml for yolov8 det model
```
model_name: "yolov8"
model_path: "path-to-onnx-model"
framework: "ONNX" # ("ONNX" or "TensorRT")
input_size: [640,640]
conf_thres: 0.25
nms_thres: 0.65
```
config/yolo/yolov8_seg.yaml for yolov8 seg model
```
model_name: "yolov8_seg"
model_path: "path-to-your-onnx-model"
framework: "ONNX"
input_size: [640,640]
conf_thres: 0.25
nms_thres: 0.65
seg_size: [160, 160]
seg_channels: 32
```

#### Run
```
mkdir -p output/yolo/normal
mkdir output/yolo/normal
mkdir output/yolo/end2end
cd build
./test/yolov8_test
```

### TensorRT
#### Build
```
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_TENSORRT=ON
make
```
#### Config
config/yolo/yolov8_normal.yaml for yolov8 det model
```
model_name: "yolov8"
model_path: "path-to-tensorrt-model"
framework: "TensorRT" # ("ONNX" or "TensorRT")
input_size: [640,640]
conf_thres: 0.25
nms_thres: 0.65
```
config/yolo/yolov8_e2e.yaml for yolov8 det model
```
model_name: "yolov8"
model_path: "path-to-tensorrt-model"
framework: "TensorRT"
input_size: [640,640]
topk: 100
```
config/yolo/yolov8_seg.yaml for yolov8 seg model
```
model_name: "yolov8_seg"
model_path: "path-to-your-onnx-model"
framework: "TensorRT"
input_size: [640,640]
conf_thres: 0.25
nms_thres: 0.65
seg_size: [160, 160]
seg_channels: 32
```

#### Run
```
mkdir -p output/yolo/normal
mkdir output/yolo/normal
mkdir output/yolo/end2end
cd build
./test/yolov8_test
```
