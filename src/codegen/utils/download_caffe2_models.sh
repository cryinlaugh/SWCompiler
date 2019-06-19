#!/usr/bin/env bash
MODELS=$(cat <<EOF
lenet_mnist
resnet50
EOF
)
download_lenet_mnist()
{
    if [ ! -d lenet_mnist ]; then
        if [ -f lenet_mnist.zip ]; then
            echo "lenet_mnist.zip exist, unzip..."
            unzip lenet_mnist.zip
        else
            echo "download lenet_mnist model..."
            wget https://cloud.tsinghua.edu.cn/f/5ccafdc5f0fe4f02a461/\?dl\=1 -O lenet_mnist.zip
            echo "unzip lenet_mnist.zip..."
            unzip lenet_mnist.zip
        fi
    else
        echo "directory lenet_mnist already exist, skip"
    fi
}
download_resnet50()
{
    if [ ! -d resnet50 ]; then
        if [ -f resnet50.zip ]; then
            echo "resnet50.zip exist, unzip..."
            unzip resnet50.zip
        else
            echo "download resnet50 model..."
            wget https://cloud.tsinghua.edu.cn/f/ac524e04c37d4f67a22b/\?dl\=1 -O resnet50.zip
            echo "unzip resnet50.zip..."
            unzip resnet50.zip
        fi
    else
        echo "directory resnet50 already exist, skip"
    fi
}

if [ "$1" != "" ]; then
    if [ "$1" == "lenet_mnist" ]; then
        download_lenet_mnist
    elif [ "$1" == "resnet50" ]; then
        download_resnet50
    fi
else
    download_lenet_mnist
    download_resnet50
fi
