# Get Started (Better Version)

## Prerequsites
**OS**: Ubuntu 22.04

Ensure you have `gcc 10`, `cuda 11.3`, and `nvcc` installed.

## Install GCC 10
### Check if gcc 10 is already installed
```
gcc --version
```
### If not installed:
### 1. Add the toolchain PPA (if not already added)
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
```
### 2. Install GCC 10 and G++ 10
```
sudo apt install gcc-10 g++-10
```
### 3. Switch to GCC 10 (Permanently)
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

## Intall CUDA 11.3 and NVCC
### Make sure you already have an NVIDIA driver installed and working
```
nvidia-smi
```
### If not installed yet, install one that's compatible:
```
sudo apt install nvidia-driver-525
```
### Manually Install Only the Toolkit (No Driver)
```
sudo apt-get install cuda-toolkit-11-3
```


## Setup Virtual Environment

**Open a terminal and do the following:**

### Make sure Python 3.8 is installed
```
python3.8 --version
```
### If not installed, install it:
```
sudo apt install python3.8 python3.8-venv
```
### If the above command throws an error do the following:
```
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev wget
```
```
cd /tmp
wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz
tar -xzf Python-3.7.17.tgz
cd Python-3.7.17
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```
### 1. Create a new venv using Python 3.8
```
python3.8 -m venv myenv38
```
### 2. Activate the venv
```
source myenv38/bin/activate
```
### 3. Install Dependencies
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_points3d==1.3.0
pip install torch-scatter==2.1.1
pip install torch-sparse==0.6.14
pip install torch-points-kernels==0.6.10
pip install torch-geometric==1.7.2
pip install timm==0.9.2
pip install termcolor==2.3.0
pip install h5py==3.8.0
pip install tensorboardX==2.6
pip install numpy==1.20.3
```
### 4. Clone repo
```
git clone https://github.com/JaredWinkens/COSeg.git
```

### 5. Compile pointops
```
cd COSeg/lib/pointops2
python3 setup.py install
```
### 6. Continue from "Dataset Preperation" in "README.md"


