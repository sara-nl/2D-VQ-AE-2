```bash
pip install "git+https://github.com/sara-nl/hydra-2.0.git"
pip install "git+https://github.com/sara-nl/hydra-2.0.git#subdirectory=plugins/hydra_optuna_sweeper"
pip install "git+https://github.com/sara-nl/hydra-2.0.git#subdirectory=plugins/hydra_submitit_launcher"

git clone https://github.com/sara-nl/2D-VQ-AE-2.git
cd 2D-VQ-AE-2

conda env create -f environment.yml
pip install .
```


# ASAP usage
ASAP (https://github.com/computationalpathologygroup/ASAP) is not in `environment.yml`  
Please install it yourself.

In Snellius/Lisa, ASAP2.0 is available under `ASAP/2.0-foss-2021a-CUDA-11.3.1`, i.e.:  
`module load 2021 ASAP/2.0-foss-2021a-CUDA-11.3.1`
