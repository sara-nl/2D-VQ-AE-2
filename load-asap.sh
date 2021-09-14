#!/bin/bash

module load 2020 Miniconda3

source activate 2D-VQ-AE-2
module load ASAP

PYTHONPATH=$PYTHONPATH:~/.conda/envs/2D-VQ-AE-2/lib/python3.8/site-packages/
