import os
from functools import partial
from subprocess import run

import pandas as pd


def format_args(
        num_tasks: int,
        num_threads: int,
        entrypoint_path: str,
        camelyon_path: str,
        network: str = 'gloo',
        precision: str = '32',  # bf16 or fp32
        num_steps: int = 50,
        tcmalloc_path: str = '',
        iomp_path: str = '',
        **kwargs
) -> list[str]:  # tuple(environment variables, args):
    network_trainer_map = {
        'gloo': 'default_trainer',
        'ccl': 'ddp_trainer_intel_cpu'
    }

    return [
            *(f"{str(k).upper()}={v}" for k, v in kwargs.items()),
            f"SLURM_TASKS_PER_NODE={num_tasks}",
            f"LD_PRELOAD={':'.join([tcmalloc_path, iomp_path])}",
            "KMP_BLOCKTIME=1",
            "KMP_AFFINITY=granularity=fine,compact,1,0",
            f"CAMELYON16_PATH={camelyon_path}",
            f"OMP_NUM_THREADS={num_threads}",
            "OMP_SCHEDULE=STATIC",
            "mpirun",
            "-disable-auto-cleanup",  # so many hours wasted to find this one flag
            "python",
            f"{entrypoint_path}",
            f"trainer={network_trainer_map[network]}",
            "model/optim@model.optim_conf=adam",
            f"trainer.precision={precision}",
            "train_datamodule.train_dataloader_conf.num_workers=5",
            "trainer.accelerator=cpu",
            f"trainer.devices={num_tasks}",
            "trainer.num_nodes=1",
            "train_datamodule.train_dataloader_conf.dataset.n_patches_per_wsi=1000",
            "+trainer.num_sanity_val_steps=0",
            "+trainer.limit_val_batches=0",
            "trainer.log_every_n_steps=1",
            "trainer.max_epochs=1",
            f"+trainer.limit_train_batches={num_steps}",
            *filter(None, ("trainer/plugins=impi_environment" if network == 'ccl' else None,)),
    ]


def write_to_file(db_path: str, args: list[str], sec_per_it: float) -> None:
    # noinspection PyTypeChecker
    kvs = dict(
        kv for arg in args
        if len(
            # examples:
            # "+trainer.num_sanity_val_steps=0" -> ["num_sanity_val_steps", "0"]
            # "trainer/plugins=impi_environment" -> ["plugins", "impi_environment"]
            # trainer/foo@trainer.bar -> [""]
            (kv := arg.strip('+').replace("/", "_").replace(".", "_").replace("@", "_").split('=', maxsplit=1))
        ) == 2  # ignore single length items such as mpirun and python
    )

    ld_preload = kvs['LD_PRELOAD'].split(':')
    for elem in ld_preload:
        if 'tcmalloc' in elem:
            kvs['malloc'] = 'tcmalloc'
        elif 'iomp' in elem:
            kvs['omp'] = 'iomp'

    del kvs['LD_PRELOAD']

    result_df = pd.DataFrame.from_records([{
        'sec_per_it': sec_per_it,
        **{k.split('.')[-1]: v for k, v in kvs.items()}  # e.g. 'trainer.max_epochs' -> 'max_epochs'
    }])

    if os.path.isfile(db_path):
        result_df = pd.concat([pd.read_pickle(db_path), result_df], ignore_index=True)

    result_df.to_pickle(db_path)


def main():
    entrypoint_path = '~/2D-VQ-AE-2/vq_ae/train.py'
    tcmalloc_path = '~/2D-VQ-AE-2/libtcmalloc.so'
    iomp_path = '~/2D-VQ-AE-2/.venv/py310-AMX/lib/libiomp5.so'
    camelyon_path = '~/CAMELYON16/'

    db_path = './results_transposedconv.df'

    f_args = partial(
        format_args,
        entrypoint_path=entrypoint_path,
        camelyon_path=camelyon_path,
        tcmalloc_path=tcmalloc_path,
        iomp_path=iomp_path
    )

    icx_run = (
        # {'num_tasks': 1, 'num_threads': 36, 'nodetype': 'icx'},
        # {'num_tasks': 1, 'num_threads': 72, 'nodetype': 'icx'},
        {'num_tasks': 2, 'num_threads': 35, 'network': 'ccl', 'nodetype': 'icx'},
        {'num_tasks': 2, 'num_threads': 36, 'network': 'ccl', 'nodetype': 'icx'},
        {'num_tasks': 4, 'num_threads': 17, 'network': 'ccl', 'nodetype': 'icx'},
        {'num_tasks': 4, 'num_threads': 18, 'network': 'ccl', 'nodetype': 'icx'},
        {'num_tasks': 8, 'num_threads': 9, 'network': 'ccl', 'nodetype': 'icx'},
        {'num_tasks': 8, 'num_threads': 8, 'network': 'ccl', 'nodetype': 'icx'},
    )

    # SPR_FP32_RUN = (
    #     {'num_tasks': 1, 'num_threads': 72},
    #     {'num_tasks': 2, 'num_threads': 36},
    #     {'num_tasks': 4, 'num_threads': 18},
    #     {'num_tasks': 8, 'num_threads': 9},
    # )

    n_runs = 5
    max_failures = 10

    for kwargs in icx_run:
        args = f_args(num_steps=50, **kwargs)
        print(args)

        for run_n in range(n_runs):
            output = run(' '.join(args), shell=True, capture_output=True)

            try:
                # god help me if the TQDM progress bar ever gets changed
                sec_per_it = str(output.stdout, 'utf-8').splitlines()[-1].split(', ')[1]
            except IndexError:
                print("===================== FAILED =====================")
                print(f"stdout: {output.stdout}, stderr: {output.stderr}")

                max_failures -= 1
                if max_failures == 0:
                    exit(1)
                continue

            sec_per_it = (
                1/float(sec_per_it.strip('it/s'))
                if 'it/s' in sec_per_it
                else float(sec_per_it.strip('s/it'))
            )

            print(f'run_n: {run_n}, n_tasks: {kwargs["num_tasks"]}, sec_per_it: {sec_per_it}')
            write_to_file(db_path, args, sec_per_it)

if __name__ == '__main__':
    main()
