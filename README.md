# CXX-DeepLearning-Inference

## Introduction
A unified and extensible pipeline for deep learning model inference with C++.
### Support framework
- [x] ONNXRuntime
- [x] TensorRT
- [x] RKNN
### Support model
- [x] object-detection
  - [x] [yolo](/doc/model/yolo.md) (including yolov8 & yolov9 for detection, segmentation and pose) 
- [x] ocr
  - [x] [crnn](/doc/model/crnn.md)
  - [x] [abinet](/doc/model/abinet.md)
  - [x] [dbnet](/doc/model/dbnet.md)
- [x] [sam](/doc/model/sam.md) 
- [x] [clip](/doc/model/clip.md)

|       |         | ONNXRuntime | TensorRT | RKNN |
|-|-|:-:|:-:|:-:|
| YOLO  | YOLO-Det|       √     |   √      |      |
|       | YOLO-Seg|       √     |   √      |      |
|       | YOLO-Pose|       √     |   √      |      |
|       | YOLO-Det-Cutoff |||       √     |
|       | YOLO-Seg-Cutoff |||       √     | 
|OCR    | CRNN    |       √     |   √      |      |
|       | ABINet  |       √     |   √      |      |
|       | DBNet   |       √     |   √      |      |
|SAM    |         |       √     |   √      |      |
|CLIP   |         |       √     |   √      |      |

## Appendix
[How to build TensorRT environment](/doc/environment/cuda-on-linux.md)
