source ./load-asap.sh
source ./.venv/py310-AMX/bin/activate
source $(python -c "import oneccl_bindings_for_pytorch as torchCCL; print(f'{torchCCL.__file__[:-12]}/env/setvars.sh')")

