#!/bin/bash

echo predeploy > /home/ec2-user/predeploy
pip3 install gdown
gdown https://drive.google.com/uc?id=1Ed2D_BNawuAQsRscIsbu_rPIm0KdJnpL -O finetune/model/