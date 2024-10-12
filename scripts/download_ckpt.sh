#!/bin/bash

mkdir -p ckpt

cd ckpt
wget https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/ckpt.zip
unzip ckpt.zip
rm ckpt.zip
cd ..

echo "Download checkpoint models finished!"
