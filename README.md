# 2D-VQ-AE-2
2D Vector-Quantized Auto-Encoder for compression of Whole-Slide Images in Histopathology

# How to run
## Locally
```bash
python train.py train_datamodule.train_dataloader_conf.dataset.path=<camelyon_path> train_datamodule.val_dataloader_conf.dataset.path=<camelyon_path>
```
## Lisa
set `CAMELYON16_PATH`, and append `--multirun` to automatically submit a `sbatch` job through `submitit`, which copies CAMELYON16 to `$SCRATCH` of the allocated node.
```bash
CAMELYON16_PATH=<camelyon-path> python train.py --multirun
```

Change node type by overwriting the node config, e.g.:
```bash
python train.py <dataset-path-args> hydra/launcher/node@hydra.launcher=gpu_titanrtx --multirun
```



## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">

