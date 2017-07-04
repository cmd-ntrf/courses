#!/bin/bash
# This script is designed to work with Calcul Québec Helios server
set -e

# ensure minimum module are loaded
module purge -f
module load compilers/gcc/4.8.5 apps/buildtools apps/git libs/hdf5

# Load GPU drivers
module load cuda/8.0.44

# Load Python 3
module load apps/python/3

# Create your own Python kernel for the course
virtualkernel deeplearning
KERNEL_PATH=$HOME/.local/share/jupyter/kernels/deeplearning/env
source $KERNEL_PATH/bin/activate

# install python science stack
pip install numpy
pip install /software6/apps/python/wheelhouse/gcc/scipy-0.19.1-cp35-cp35m-linux_x86_64.whl
pip install matplotlib
pip install bcolz
pip install h5py

# Install libgpuarray
(
pip install cython
module load apps/cmake/3.4.0
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR
git clone https://github.com/Theano/libgpuarray
cd libgpuarray
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=$KERNEL_PATH ../
make install
cd ..
python setup.py build_ext -I $KERNEL_PATH/include -L $KERNEL_PATH/lib
python setup.py install
cd
rm -rf $TEMP_DIR
)

# install and configure theano
pip install theano
echo "[global]
device = gpu
floatX = float32

[cuda]
root = $CUDA_HOME" > ~/.theanorc

# install and configure keras
pip install keras==1.2.2
mkdir -p ~/.keras
echo '{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}' > ~/.keras/keras.json

# install cudnn libraries
module load libs/cuDNN/6

module save fast.ai
