# CUDA ON Ubuntu

## 驱动安装

TODO

## cuda安装

### 添加源

For Ubuntu 22.04
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
sudo sh -c 'echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
```

### 安装
```
# 更新列表
sudo apt-get update

# 查询可用版本
apt search cuda-toolkit

# 安装
sudo apt install cuda-toolkit-<version>
```

## TensorRT安装

```
# 查询可用版本
apt policy tensorrt-dev

# 安装
sudo apt install tensorrt-dev=<version>

# 安装trtexec
sudo apt install libnvinfer-bin=<version>
```