#!/usr/bin/env bash
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# caffe install - protobuf, opencv, leveldb
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

# boost install
sudo apt-get install --no-install-recommends libboost-all-dev

# blas install
sudo apt-get install libatlas-base-dev

# python-dev install
sudo apt-get install the python-dev

# remaining dependencies
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

# pytube
pip install -r requirements.txt

# git
sudo apt-get install git

# update pip
pip install --upgrade pip

# opencv
sudo apt-get install python-opencv
conda install opencv

# config git
git config --global user.email "ioana.chelu28@gmail.com"
git config --global user.name "ioana chelu"
git config --global push.default matching
