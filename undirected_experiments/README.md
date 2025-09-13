
# LRGB evaluation
Evaluation taken from (https://github.com/toenshoff/LRGB/tree/main) which is based on the GPS codebase: https://github.com/rampasek/GraphGPS

### Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


### Running Training
Training configuration handled by config files in `configs` directory. As an example, you can run experiments with the following commands.
```bash
conda activate graphgps
python main.py --cfg configs/vocsuperpixels_UnitaryGCN_enhanced.yaml wandb.use False
python main.py --cfg configs/zinc_UnitaryGCN_enhanced.yaml wandb.use False
```
If setting `wandb.use` to True, add an entity username in the config file before doing so.