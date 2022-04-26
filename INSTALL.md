# Install PDM
e.g:
- Linux/Mac: `curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -`
- Windows: `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py -UseBasicParsing).Content | python -`

# Install repo/dependencies
```bash
git clone https://github.com/sara-nl/2D-VQ-AE-2.git
cd 2D-VQ-AE-2

pdm install
```


# ASAP usage
ASAP (https://github.com/computationalpathologygroup/ASAP) is not in `pyproject.toml`  
Please install it yourself.

In Snellius/Lisa, ASAP2.0 is available under `ASAP/2.0-foss-2021a-CUDA-11.3.1`, i.e.:  
`module load 2021 ASAP/2.0-foss-2021a-CUDA-11.3.1`

Load the environment easily by executing `source ./load-asap.sh` after following the previous installation steps.