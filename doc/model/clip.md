# Clip


## Related repos
- [Clip](https://github.com/openai/CLIP)
- [onnx_clip](https://github.com/lakeraai/onnx_clip.git)
- [clip-distillation](https://github.com/NVIDIA-AI-IOT/clip-distillation)

## ONNXRuntime Inference
### Model
You can download the ONNX model by
```
wget https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_image_model_vitb32.onnx
wget https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_text_model_vitb32.onnx
```
Or you can export by yourself. Python code will be like
```
torch.onnx.export(image_model,
                torch.randn(1, 3, 224, 224),
                f,
                opset_version=11,
                input_names=['IMAGE'],
                output_names=['IMAGE_EMBEDDING'],
                dynamic_axes={
                    'IMAGE': {0: 'batch_size'},
                    'IMAGE_EMBEDDING': {0: 'batch_size'}
                })

torch.onnx.export(text_model,
                torch.randn(1, 77),
                f,
                opset_version=11,
                input_names=['TEXT'],
                output_names=['TEXT_EMBEDDING'],
                dynamic_axes={
                    'TEXT': {0: 'batch_size'},
                    'TEXT_EMBEDDING': {0: 'batch_size'}
                })
```

### Inference
#### Build
```
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_TENSORRT=OFF
make
```
#### Config
config/clip/image_encoder.yaml
```
model_name: "clip_image_encoder"
model_path: "../weights/clip/clip_image_model_vitb32.onnx"
framework: "ONNX"
```
config/clip/text_encoder.yaml
```
model_name: "clip_text_encoder"
model_path: "../weights/clip/clip_text_model_vitb32.onnx"
framework: "ONNX"
bpe_path: "../weights/clip/bpe_simple_vocab_16e6.txt.gz"
```
#### Run
```
cd build
./test/clip_test
```
You can get output like
```
Input: 
IMAGE: [-1,3,224,224,]
Output: 
IMAGE_EMBEDDING: [-1,512,]
Input: 
TEXT: [-1,77,]
Output: 
TEXT_EMBEDDING: [-1,512,]
Shape of IMAGE_EMBEDDING: [2,512,]
Shape of TEXT_EMBEDDING: [2,512,]
[ [ 0.970533 0.0294665  ], [ 0.0195933 0.980407  ],  ]
Destruct text encoder
Destruct image encoder
```

## TensorRT Inference
### Model
We can simply transfer the onnx model to tensorrt engine by
```
/usr/src/tensorrt/bin/trtexec \
    --onnx=${onnx_image_model_path} \
    --saveEngine=${tensorrt_image_model_path} \
    --fp16 \
--minShapes=IMAGE:1x3x224x224 \
--optShapes=IMAGE:1x3x224x224 \
--maxShapes=IMAGE:10x3x224x224
/usr/src/tensorrt/bin/trtexec \
    --onnx=${onnx_text_model_path} \
    --saveEngine=${tensorrt_text_model_path} \
    --fp16 \
--minShapes=TEXT:1x77 \
--optShapes=TEXT:1x77 \
--maxShapes=TEXT:10x77
```
However, these models with vit is too large for Jetson.

Nvidia releases [clip-distillation](https://github.com/NVIDIA-AI-IOT/clip-distillation) to solve this problem. 
First, train a smaller image model with knowledge distillation.
Though Nvidia does not release weights yet, it publishes a pipeline to train models by our own.
Besides, fix the prompt texts and save their embeddings.

### Inference
#### Build
```
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_TENSORRT=ON
make
```
#### Config
config/clip/image_encoder.yaml
```
model_name: "clip_image_encoder"
model_path: "../weights/clip/clip_image_model_res18.engine"
framework: "TensorRT"
```
#### Run
```
cd build
./test/clip_test
```