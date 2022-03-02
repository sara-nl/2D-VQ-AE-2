```bash
git clone https://github.com/sara-nl/2D-VQ-AE-2.git
cd 2D-VQ-AE-2

conda env create -f environment.yml
conda activate 2D-VQ-AE-2

pip install .

pip install "git+https://github.com/sara-nl/hydra-2.0.git"
pip install "git+https://github.com/sara-nl/hydra-2.0.git#subdirectory=plugins/hydra_optuna_sweeper"
pip install "git+https://github.com/sara-nl/hydra-2.0.git#subdirectory=plugins/hydra_submitit_launcher"
```


# ASAP usage
ASAP (https://github.com/computationalpathologygroup/ASAP) is not in `environment.yml`  
Please install it yourself.

In Snellius/Lisa, ASAP2.0 is available under `ASAP/2.0-foss-2021a-CUDA-11.3.1`, i.e.:  
`module load 2021 ASAP/2.0-foss-2021a-CUDA-11.3.1`

Concurrent usage of modules and conda has some nuances regarding the `$PYTHONPATH` environment variable.  
For easy usage of the provided conda environment on Snellius/Lisa, the following alias can be added to `.bashrc` or `.bash_profile`:
```bash
alias load-env="module load 2021 Miniconda3/4.9.2 ASAP/2.0-foss-2021a-CUDA-11.3.1; conda activate 2D-VQ-AE-2; export PYTHONPATH=~/.conda/envs/2D-VQ-AE-2/lib/python3.9/site-packages:\$PYTHONPATH"
```
After which the environment can be easily loaded by running `load-env` from the command line.
