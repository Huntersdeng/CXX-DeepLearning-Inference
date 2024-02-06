# ONNXRuntime install

1. choose a version in [Onnxruntime release](https://github.com/microsoft/onnxruntime/releases)
2. download
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.2/onnxruntime-linux-x64-1.16.2.tgz
```
3. install
```
tar -xzf onnxruntime-linux-x64-1.16.2.tgz
cd onnxruntime-linux-x64-1.16.2
sudo cp include/* /usr/include
sudo cp lib/* /usr/lib
```