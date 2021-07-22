from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from conf.preprocessing.datamodule import CAMELYON16DataloaderConf, CAMELYON16DatasetConf, CAMELYON16DataModuleConf
from utils.train_helpers import Stage


if __name__ == '__main__':

    TEST_TRAIN_DATASET = CAMELYON16DatasetConf(path='/project/robertsc/examode/CAMELYON16/')
    TEST_VAL_DATASET = CAMELYON16DatasetConf(path='/project/robertsc/examode/CAMELYON16/', train=Stage.VALIDATION)
    TEST_DATAMODULE = CAMELYON16DataModuleConf(
        train_dataloader_conf=CAMELYON16DataloaderConf(dataset=TEST_TRAIN_DATASET),
        val_dataloader_conf=CAMELYON16DataloaderConf(dataset=TEST_VAL_DATASET),
    )

    cs = ConfigStore.instance()
    cs.store(
        name='test_camelyon16_datamodule',
        node=TEST_DATAMODULE
    )

    with initialize(config_path=None, job_name="test_datamodule"):
        cfg = compose(config_name="test_camelyon16_datamodule")

    datamodule = instantiate(cfg)
    datamodule.setup()
    
    train_datapoint = next(iter(datamodule.train_dataloader()))
    validation_datapoint = next(iter(datamodule.train_dataloader()))


    # data_dir = Path()

    # # Dataset test
    # parser = ArgumentParser()
    # parser = CAMELYON16RandomPatchDataSet.add_dataset_specific_args(parser)
    # config = parser.parse_args(['--data-dir', str(data_dir)])

    # dataset = CAMELYON16RandomPatchDataSet(config)

    # # check for tissue
    # datapoint_index, n_tries = 1, 1
    # n_success = sum(
    #     np.any(np.logical_or(dataset[datapoint_index] > 0, dataset[datapoint_index] < 255))
    #     for _ in range(n_tries)
    # )
    # print(f'succes: {n_success}/{n_tries}')
