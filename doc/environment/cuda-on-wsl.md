# CUDA ON WSL

## 驱动安装

[CUDA on WSL 驱动下载地址](https://developer.nvidia.com/cuda/wsl)

根据自己的GPU类型（GeForce and Quadro) 选择对应的驱动。
不需要在wsl下安装nvidia驱动，windows会自动为wsl安装nvidia驱动。

## cuda安装

### 添加源

For Ubuntu 22.04
```
sudo apt-key adv --fetch-keys https://mirrors.aliyun.com/nvidia-cuda/ubuntu2204/x86_64/3bf863cc.pub
sudo sh -c 'echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
```

For Ubuntu 20.04
```
sudo apt-key adv --fetch-keys https://mirrors.aliyun.com/nvidia-cuda/ubuntu2004/x86_64/3bf863cc.pub
sudo sh -c 'echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
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