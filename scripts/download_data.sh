#!/bin/bash

mkdir -p data

cd data
wget https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/data.zip
unzip data.zip
rm data.zip
cd ..

echo "Download data finished!"
