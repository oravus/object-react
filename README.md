# ObjectReact: Learning Object-Relative Control for Visual Navigation [CoRL 2025]

This repository is based on [GNM](https://github.com/robodhruv/visualnav-transformer) for training an object-relative controller, dubbed [ObjectReact](https://object-react.github.io/).

### Environment

<details>
  <summary> Setup Conda environment (without habitat).
  
  (see [this](https://github.com/oravus/object-rel-nav?tab=readme-ov-file#environment) to setup with habitat) 
  </summary>

```
conda create -n nav
conda activate nav

conda install python=3.9 mamba -c conda-forge
mamba install pip numpy matplotlib pytorch torchvision pytorch-cuda=11.8 opencv=4.6 cmake=3.14.0 numba=0.57 pyyaml ipykernel networkx h5py natsort transformers einops scikit-learn kornia pgmpy python-igraph pyvis -c pytorch -c nvidia -c conda-forge
mamba install -c conda-forge tyro faiss-gpu scikit-image ipykernel spatialmath-python gdown utm seaborn wandb kaggle yacs
```
</details>

### Data

```
cd train/vint_train/data/data_splits/
huggingface-cli download oravus/objectreact_hm3d_iin training/bigger_bot_0.3-sh_0.4.zip --repo-type dataset --local-dir ./
unzip -q training/bigger_bot_0.3-sh_0.4.zip -d training/
rm -r training/bigger_bot_0.3-sh_0.4.zip
```

### Train

```
conda activate nav
pip install -e train/
cd train/
python train.py -c config/object_react.yaml
```

### Test
Please refer to the [object-rel-nav](https://github.com/oravus/object-rel-nav) repository for testing and benchmarking.

### Cite
```
@inproceedings{garg2025objectreact,
  title={ObjectReact: Learning Object-Relative Control for Visual Navigation},
  author={Garg, Sourav and Craggs, Dustin and Bhat, Vineeth and Mares, Lachlan and Podgorski, Stefan and Krishna, Madhava and Dayoub, Feras and Reid, Ian},
  booktitle={Conference on Robot Learning},
  year={2025},
  organization={PMLR}
}
```