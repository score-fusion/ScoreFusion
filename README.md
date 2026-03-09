# Score-Fusion

<p align="center">
  <b>Official PyTorch Implementation</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.07430"><img src="https://img.shields.io/badge/arXiv-2501.07430-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/xiyuez2/med_palette_2D/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

> **Introducing 3D Representation for Medical Image Volume-to-Volume Translation via Score Fusion**
>
> [Xiyue Zhu]()\*, [Dou Hoon Kwark]()\*, [Ruike Zhu](), [Kaiwen Hong](), [Yiqi Tao](), [Shirui Luo](), [Yudu Li](), [Zhi-Pei Liang](), [Volodymyr Kindratenko]()
>
> We propose Score-Fusion, which effectively learns 3D representations by ensembling perpendicularly trained 2D diffusion models in score function space. Our approach reduces computational demands through strategic model initialization and hierarchical layer design, and extends to multi-modality scenarios through diffusion model fusion.

> This repo also supports various baselines, including TPDM, TOSM, MADM, 2D Palette, Med-DDPM (e.g. 3D palette).

---

## Overview

ScoreFusion consists of two main components:

| Component | Description |
|-----------|-------------|
| `med_palette_2D` | 2D diffusion model based on Palette for slice-wise super-resolution |
| `3D_scorefuser` | 3D integration module that fuses 2D priors from three orthogonal directions |

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.13+
- CUDA 11.6+

### Setup Environment

```bash
conda create -n scorefusion python=3.9
conda activate scorefusion
```

### Dependencies

**Core dependencies:**
```
torch, torchvision, numpy, pandas, tqdm, tensorboardX, scipy, opencv-python, clean-fid
```

**Additional dependencies for 3D processing:**
```
torchio, nibabel
```

---

## Data Preparation

### Dataset

We use the [BraTS 2021](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863) dataset for training and evaluation.

### Directory Structure and Preparation

Set up symbolic links to connect the two modules:

```bash
# In med_palette_2D directory
cd med_palette_2D
ln -s /path/to/BraTs2021/ Brast21
ln -s ../3D_scorefuser/dataset_brats.py dataset_brats.py
ln -s ../3D_scorefuser/dataset_brats.py ./data/util/dataset_brats.py
cd ..

# In 3D_scorefuser directory
cd 3D_scorefuser
ln -s ../med_palette_2D/config config
ln -s ../med_palette_2D/experiments experiments
ln -s ../med_palette_2D/ model_2D
cd ..
```

Install the two directory

```bash
cd 3D_scorefuser
pip install -v -e .

cd .. 
cd med_palette_2D
pip install -v -e .
cd .. 

```

---

## Training

### Stage 1: Train 2D Diffusion Models

Train three 2D models for each orthogonal direction (axial, coronal, sagittal).

```bash
cd med_palette_2D

# Direction 1 (axial)
python run.py -p train -c config/brast_5SSR_1.json

# Direction 2 (coronal)
python run.py -p train -c config/brast_5SSR_2.json

# Direction 3 (sagittal)
python run.py -p train -c config/brast_5SSR_3.json
```

<details>
<summary><b>Configuration Options</b></summary>

| Type | Description |
|------|-------------|
| `SR` | super-resolution without nearby_slices |
| `5SSR` | 5-slice super-resolution (nearby_slices=2) |
| `2cond` | Dual-condition model (flair_poolx4 + t1ce → flair) |
| `5S2cond` | 5-slice + dual-condition |
| `MT` | modality-translation without nearby_slices |
| `5SMT` | 5-slice modality-translation |

</details>

<details>
<summary><b>Key Config Parameters</b></summary>

```json
{
    "datasets": {
        "train": {
            "args": {
                "input_modality": "flair_poolx4",
                "target_modality": "flair",
                "slice_direction": 1,
                "nearby_slices": 2
            }
        }
    },
    "model": {
        "which_networks": [{
            "args": {
                "unet": {
                    "in_channel": 10,
                    "inner_channel": 64
                },
                "beta_schedule": {
                    "train": { "n_timestep": 2000 }
                }
            }
        }]
    }
}
```

The input/target modality format in the config files follows this pattern: <modality>_<downsample_method>x<downsample_ratio>. To customize your own modalities, look at 3D_scorefuser/dataset_brats.py.

</details>

### Stage 2: Train 3D Fusion Model in two phases

After 2D models are trained, configure their checkpoints in the same json file (This will be used in 3D training as well):

```json
// In config/brast_5SSR_[1,2,3].json
"path": {
    "resume_state": "experiments/train_SR_brast_2D_X_XXXXXX/checkpoint/xxx"
}
```



| Phase | `--baseline` | Crop Parameters | Description |
|-------|--------------|-----------------|-------------|
| 1 | `3D` | `--random_crop_xy 64 --random_crop_z 64` | Train with patches |
| 2 | `3D_feature` | None | Fine-tune on full volume |



**Phase 1: Training with random crops**

```bash
cd 3D_scorefuser

python train_ensemble_325model.py \
    --dataset_folder Brast21 \
    --input_modality "flair_poolx4" \
    --target_modality "flair" \
    --res_folder "results" \
    --gpus 1 --fp16 \
    --glob_pos_emb --none_zero_mask --residual_training \
    --num_channels 64 --num_res_blocks 2 \
    --batchsize 1 --epochs 1000000 --timesteps 2000 \
    --save_and_sample_every 1000 \
    --with_condition --gradient_accumulate 4 \
    --num_workers 4 \
    --random_crop_xy 64 --random_crop_z 64 \
    --model_2D_1_config "config/brast_5SSR_1.json" \
    --model_2D_2_config "config/brast_5SSR_2.json" \
    --model_2D_3_config "config/brast_5SSR_3.json" \
    --baseline "3D"
```

**Phase 2: Fine-tuning on full volume**

```bash
python train_ensemble_325model.py \
    --dataset_folder Brast21 \
    --input_modality "flair_poolx4" \
    --target_modality "flair" \
    --res_folder "results" \
    --gpus 1 --fp16 \
    --glob_pos_emb --none_zero_mask --residual_training \
    --num_channels 64 --num_res_blocks 2 \
    --batchsize 1 --epochs 1000000 --timesteps 2000 \
    --save_and_sample_every 1000 \
    --with_condition --gradient_accumulate 4 \
    --num_workers 4 \
    --model_2D_1_config "config/brast_5SSR_1.json" \
    --model_2D_2_config "config/brast_5SSR_2.json" \
    --model_2D_3_config "config/brast_5SSR_3.json" \
    --baseline "3D_feature" \
    --zero_init_feature
```


## Inference

```bash
python train_ensemble_325model.py \
    --dataset_folder /path/to/dataset/Brast21 \
    --input_modality "flair_poolx4" \
    --target_modality "flair" \
    --res_folder "results" \
    --resume_weight /path/to/checkpoint/SR_MADM.pt \
    --gpus 1  --fp16 \
    --glob_pos_emb  --none_zero_mask --residual_training \
    --validation  --fast_sample \
    --num_channels 64  --num_res_blocks 2 \
    --batchsize 1  --epochs 1000000  --timesteps 2000 \
    --save_and_sample_every 1000 \
    --with_condition  --gradient_accumulate 1 \
    --num_workers 4 \
    --model_2D_1_config "config/brast_5SSR_1_resume.json" \
    --model_2D_2_config "config/brast_5SSR_2_resume.json" \
    --model_2D_3_config "config/brast_5SSR_3_resume.json" \
    --baseline "3D_feature" \
    --zero_init_feature
```


---

## Development Hints and Other Configs
ScoreFusion uses a 3D network to fuse 2D scores. Core logic is simple and in 3D_scorefuser/diffusion_model/unet_brats.py. 

Different combinations of 3D and 2D networks can lead to meaningful re-implementation of baselines on Brats. 
Several different main python files use different classes in unet_brats.py and can use different combinations of 2D nets. The parameter "--baseline" indicates how the 2D scores are fused. 


Python files settings:

| Python scripts |  Description |
|----------------|--------------------------------------------------|
| `train_ensemble.py` | Two 2D models along slice_direction 2 and 3 |
| `train_ensemble_3model.py` | Three 2D models along all slice_directions |
| `train_ensemble_325model.py` | Three 2.5D models, which use 5 nearby slices from the condition input |

`--baseline` options and meanings:

| `--baseline` |  Description |
|----------------|--------------------------------------------------|
| `2D` | Use one of the 2D models to directly output a slice-wise result |
| `TPDM` | Use a random 2D model at each diffusion step |
| `merge` | Uses the average of all 2D models |
| `3D_only` | Do not use 2D models, only use a 3D model to directly output the score |
| `3D` | Use a 3D network to ensemble 2D models |
| `3D_feature` | Same as `3D`, but introduces 2D feature maps into the 3D network |
| `dummy` | Output noise for debug purpose |

Several combinations of the above result in reasonable re-implementations of baselines on Brats:

| baseline methods |  Setting | 2D training | 3D training |
|----------------|--------------------------------------------------|---|---|
| TPDM | use `train_ensemble.py` with `TPDM` | Need to train 2D models without nearby_slices  | None |
| TOSM | use `train_ensemble_3model.py` with `merge` | Need to train 2D models without nearby_slices | None |
| MADM | use `train_ensemble_325model.py` with `merge` | Need to train 2D models with nearby_slices | None |
| 3D Palette | use `train_ensemble_325model.py` with `3D_only` | None | Need to train 3D model |
| 2D palette | use `train_ensemble.py` with `2D` | Need to train 2D models without nearby_slices | None |
| 2.5D palette | use `train_ensemble_325model.py` with `2D` | Need to train 2D models with nearby_slices | None |

Note that for all baselines, you need to add `--no_self_consistency` flag during inference. This results in a more reasonable implementation of baselines and slightly worse performance. Also, to utilize pretrained 2D models, "resume_state" in the 2D config needs to be changed and the 2D config file names need to be passed into the 3D/inference stage.

The training logic is in 3D_scorefuser/diffusion_model/trainer_brats.py. For better visualization, the first two GPUs will plot samples in the training set and the other GPUs will plot validation samples. 

Current resume strategy did not reuse the optimizer parameters. This works fine with cross-stage pretraining + fine-tuning in the current repo. If you want to resume within the same stage, please look at the save and resume functions in the code and make adjustments accordingly.

Unfortunately, we lost all ckpts due to a system update on the cluster. Therefore, we have no plan in releasing the pretrained checkpoints.

## Results

Results are saved in the following structure:

```
experiments/
└── train_SR_brast_2D_1_XXXXXX/
    ├── checkpoint/      # Model checkpoints
    └── results/         # Validation outputs
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhu2025score,
    title={Introducing 3D Representation for Medical Image Volume-to-Volume Translation via Score Fusion},
    author={Zhu, Xiyue and Kwark, Dou Hoon and Zhu, Ruike and Hong, Kaiwen and Tao, Yiqi and Luo, Shirui and Li, Yudu and Liang, Zhi-Pei and Kindratenko, Volodymyr},
    journal={ICML},
    year={2025}
}
```

---

## Acknowledgements

This codebase builds upon:
- [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- [med-ddpm](https://github.com/mobaidoctor/med-ddpm)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
