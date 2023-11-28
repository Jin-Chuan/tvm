#!/bin/bash

db=(densenet201 resnet152)
for net in ${db[@]}
do
python3 recorder.py $net
done