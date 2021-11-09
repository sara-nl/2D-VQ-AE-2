#!/bin/bash

module load 2020
module load Miniconda3/4.7.12.1

module load ASAP/8c9a8fb-fosscuda-2020a

source activate 2D-VQ-AE-2

export PYTHONPATH=$PYTHONPATH:~/.conda/envs/2D-VQ-AE-2/lib/python3.8/site-packages/
