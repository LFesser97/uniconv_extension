import os
import uuid
import numpy as np
import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch_geometric.loader import NeighborSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.utils.utils import use_best_hyperparams, get_available_accelerator
from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model import get_model, LightingFullBatchModelWrapper
from src.utils.arguments import args


def run(args):
    if not hasattr(args, "scheduler_t_max") or args.scheduler_t_max is None:
        args.scheduler_t_max = args.num_epochs      # <<< Default T_max to total epochs
    if not hasattr(args, "scheduler_eta_min") or args.scheduler_eta_min is None:
        args.scheduler_eta_min = 0.0                # <<< Default eta_min to 0.0
        
    torch.manual_seed(0)

    # Load dataset
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    try:
        data = dataset._data
    except AttributeError:
        data = dataset.data

    data_loader = DataLoader(
        FullBatchGraphDataset(data),
        batch_size=8,
        collate_fn=lambda batch: batch[0]
    )

    val_accs, test_accs = [], []
    for run_idx in range(args.num_runs):
        # WandB
        run_name = f"{args.dataset}_{run_idx}"
        wandb_logger = WandbLogger(log_model=False, name=run_name)

        # Splits
        train_mask, val_mask, test_mask = get_dataset_split(
            args.dataset, data, args.dataset_directory, run_idx
        )

        # Build model
        args.num_features, args.num_classes = data.num_features, dataset.num_classes
        model = get_model(args)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            scheduler_t_max=args.scheduler_t_max,
            scheduler_eta_min=args.scheduler_eta_min
        )

        # Callbacks
        early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
        summary    = ModelSummary(max_depth=-1)
        os.makedirs(args.checkpoint_directory, exist_ok=True)
        ckpt = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=os.path.join(args.checkpoint_directory, str(uuid.uuid4())),
        )

        # Trainer (AMP enabled)
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[early_stop, summary, ckpt],
            profiler="simple" if args.profiler else None,
            accelerator=get_available_accelerator(),
            devices=[args.gpu_idx],
            precision=32,
            logger=wandb_logger,
        )

        # Fit & test
        trainer.fit(model=lit_model, train_dataloaders=data_loader)
        val_acc = ckpt.best_model_score.item()
        test_res = trainer.test(ckpt_path="best", dataloaders=data_loader)
        test_acc = test_res[0].get("test_acc")

        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(f"Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")


if __name__ == "__main__":
    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    run(args)