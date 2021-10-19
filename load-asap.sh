#!/bin/bash

source ~/.bashrc

module load 2020 Miniconda3

source deactivate
source deactivate

module load ASAP

source activate 2D-VQ-AE-2

export PYTHONPATH=$PYTHONPATH:~/.conda/envs/2D-VQ-AE-2/lib/python3.8/site-packages/
