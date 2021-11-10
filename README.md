# 2D-VQ-AE-2
2D Vector-Quantized Auto-Encoder for compression of Whole-Slide Images in Histopathology

## How to run

### Installation
See [INSTALL.md](https://github.com/sara-nl/2D-VQ-AE-2/blob/main/INSTALL.md).  
We provide both an `environment.yml` and a `requirements.txt`, but we suggest the usage of `environment.yml`.

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

## WIP results
Top: original, bottom: reconstruction.  
Input dimensionality: `256×256×3` ordinal 8-bit, latent dimensionality: `32×32` categorical 8-bit (i.e. `99.47%` compression).

![image](https://user-images.githubusercontent.com/5969044/134488209-4c1696d3-6478-41d0-a7bf-e7e99544382b.png)
![image](https://user-images.githubusercontent.com/5969044/134643133-26268fed-d950-4441-82f0-a2358c9d114d.png)



## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">

