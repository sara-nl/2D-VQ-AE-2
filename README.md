# 2D-VQ-AE-2
2D Vector-Quantized Auto-Encoder for compression of Whole-Slide Images in Histopathology

## How to run

### Installation
See [INSTALL.md](./INSTALL.md).  
We use PDM as python package manager (https://github.com/pdm-project/pdm).

### Locally
set `CAMELYON16_PATH` and run `train.py`:
```bash
CAMELYON16_PATH=<camelyon-path> python train.py
```
### Lisa
set `CAMELYON16_PATH`, and append `--multirun` to automatically submit a `sbatch` job through `submitit`.  
- If `CAMELYON16_PATH` is a folder, the dataloader loads the dataset over the network.
- If `CAMELYON16_PATH` is a `.tar`, the file is copied to `$SCRATCH` of the allocated node, and the dataset is loaded locally.
```bash
CAMELYON16_PATH=<camelyon-path> python train.py --multirun
```
Change node type by overwriting the node config, e.g.:
```bash
CAMELYON16_PATH=<camelyon-path> python train.py hydra/launcher/node@hydra.launcher=gpu_titanrtx --multirun
```

## Results
Note on Mean Squared-Error results: input is channel-wise normalised to 0-mean, 1-std, using the following values, based on 10k patches:
|  | Red    | Green  | Blue   |
|------------------------|--------|--------|--------|
| Mean                   | 0.7279 | 0.5955 | 0.7762 |
| Standard Deviation     | 0.2419 | 0.3083 | 0.1741 |

Top: original, bottom: reconstruction.  
Input dimensionality: `256×256×3@0.5μm` ordinal 8-bit, latent dimensionality: `32×32@16μm` categorical 8-bit (i.e. `99.47%` compression), `0.900 MSE`.

![image](https://user-images.githubusercontent.com/5969044/134488209-4c1696d3-6478-41d0-a7bf-e7e99544382b.png)
![image](https://user-images.githubusercontent.com/5969044/134643133-26268fed-d950-4441-82f0-a2358c9d114d.png)

Input dimensionality: `512×512×3@0.25μm` ordinal 8-bit, latent dimensionality: `32×32@16μm` categorical 8-bit (i.e. `99.87%` compression), `0.800 MSE`.

![9233614](https://user-images.githubusercontent.com/5969044/171684798-b0bc1242-1941-4dd5-bfc7-d44bb9c59024.png)
![9233614_2](https://user-images.githubusercontent.com/5969044/171684803-43d1473e-b479-4f3e-a698-cb1f48b1bc74.png)


## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">

