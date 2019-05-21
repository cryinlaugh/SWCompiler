#!/usr/bin/env bash
if [ ! -f mnist_labels_images.tar.gz ]; then
    echo "download from https://cloud.tsinghua.edu.cn/f/f78dc973eb2f4195a829/?dl=1"
    wget https://cloud.tsinghua.edu.cn/f/f78dc973eb2f4195a829/?dl=1 -O mnist_labels_images.tar.gz
fi

if [ ! -f cifar-10-binary.tar.gz ]; then
    echo "download from https://cloud.tsinghua.edu.cn/f/7fea9398293c4c859a97/?dl=1" 
    echo "mirror of http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    wget https://cloud.tsinghua.edu.cn/f/7fea9398293c4c859a97/?dl=1 -O cifar-10-binary.tar.gz
fi

