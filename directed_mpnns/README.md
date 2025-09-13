### Setting Up the Environment

This part of the project is designed to run on Python 3.10. We recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to set up the environment as follows:

```bash
conda create -n directed_gnn python=3.10
conda activate directed_gnn
```

### Installing Dependencies

Once the environment is activated, install the required packages:

```bash
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg pytorch-sparse -c pyg
pip install ogb==1.3.6
pip install pytorch_lightning==2.0.2
pip install gdown==4.7.1
```

Please ensure that the version of `pytorch-cuda` matches your CUDA version. If your system does not have a GPU, use the following command to install PyTorch:

```bash
conda install pytorch==2.0.1 -c pytorch
```

For M1/M2/M3 Mac users, `pyg` (PyTorch Geometric) needs to be installed from source. Detailed instructions for this process can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-from-source).

## Running Experiments

This section provides instructions on how to reproduce the experiments outlined in the paper. Note that some of the results may not be reproduced *exactly*, given that some of the operations used are intrinsically non-deterministic on the GPU, as explained [here](https://github.com/pyg-team/pytorch_geometric/issues/92). However, you should obtain results very close to those in the paper.


To run the experiments for the different datasets yourself, you can use the following type of command in the terminal:

```bash
python -m src.run --dataset directed-roman-empire --model dir-uni --num_runs 10 --patience 200 --normalize
```

You can further specify model hyperparameters by setting `--num_layers`, `--lr`, `--hidden_dim`, or `--dropout`.



## Acknowledgements

Much of the codebase for this part of the project is based on the repository for the following paper.

```bibtex
@misc{dirgnn_rossi_2023,
    title={Edge Directionality Improves Learning on Heterophilic Graphs},
    author={Emanuele Rossi and Bertrand Charpentier and Francesco Di Giovanni and Fabrizio Frasca and Stephan Günnemann and Michael Bronstein},
    publisher={arXiv},
    year={2023}
}
```