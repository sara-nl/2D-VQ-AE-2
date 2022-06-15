#!/bin/bash

module load 2021
module load ASAP/2.0-foss-2021a-CUDA-11.3.1
module load Python/3.9.5-GCCcore-10.3.0

module unload cuDNN # module cuDNN interferes with pdm installed cuDNN
