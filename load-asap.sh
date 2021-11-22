#!/bin/bash

module load 2021
module load Miniconda3/4.9.2

module load ASAP/2.0-foss-2021a-CUDA-11.3.1

source deactivate
source deactivate
source activate 2D-VQ-AE-2

export PYTHONPATH=$PYTHONPATH:~/.conda/envs/2D-VQ-AE-2/lib/python3.9/site-packages/
