source ./load-asap.sh
eval $(pdm venv activate py39-cpu)
source $(python -c "import oneccl_bindings_for_pytorch as torchCCL; print(f'{torchCCL.__file__[:-12]}/env/setvars.sh')")

