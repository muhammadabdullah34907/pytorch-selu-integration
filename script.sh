git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive

cp ../rnn-with-SELU/RNN.cpp aten/src/ATen/native/RNN.cpp
cp ../rnn-with-SELU/RNN.cu aten/src/ATen/native/cuda/RNN.cu

conda create --name pytorch-env -y
conda activate pytorch-env
conda install cmake ninja -y
pip3 install -r requirements.txt

conda install mkl mkl-include -y
conda install -c pytorch magma-cuda110 -y #replace 110 with your cuda version such as 118 for CUDA 11.8

export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python3 setup.py develop
