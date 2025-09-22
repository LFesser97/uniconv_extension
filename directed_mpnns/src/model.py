import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
)

from torchmetrics.classification import BinaryAUROC


from src.datasets.data_utils import get_norm_adj
from src.complex_valued_layers import UnitaryGCNConvLayer
from src.directed_layers import DirectedUnitaryGCNConvLayer


def get_conv(conv_type, input_dim, output_dim, alpha, T=20, dropout=0, residual=False):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    elif conv_type == "dir-uni":
        return DirectedUnitaryGCNConvLayer(input_dim, output_dim, dropout=dropout, residual=residual)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x
        )


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
    ):
        super(GNN, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # if i+1 in [4, 8, 16, 32, 64]:
                # print(f"Dirichlet Energy at Layer {i+1}: {dirichlet_energy(x, edge_index).item()}")
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)
    

class UnitaryGCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
    ):
        super(UnitaryGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.input_dim = num_features
        self.hidden_dim = hidden_dim
        self.norm = torch.nn.LayerNorm(self.input_dim)
        output_dim = num_classes
        self.num_layers = num_layers
        self.T = 20
        self.hermitian = False
        self.residual = False
        self.conv_layers.append(UnitaryGCNConvLayer(self.input_dim, self.hidden_dim, T = self.T, dropout = dropout))
        for _ in range(self.num_layers):
            self.conv_layers.append(UnitaryGCNConvLayer(self.hidden_dim, self.hidden_dim, use_hermitian=self.hermitian, residual = self.residual, dropout = dropout, T = self.T)) 
        
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x_in, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x_in = layer(x_in, edge_index)
        return torch.nn.functional.log_softmax(self.mlp(x_in.real), dim=1)
    
    
class DirectedUnitaryGCN(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
    ):
        super(DirectedUnitaryGCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.input_dim = num_features
        self.hidden_dim = hidden_dim
        self.norm = torch.nn.LayerNorm(self.input_dim)
        output_dim = num_classes
        self.num_layers = num_layers
        self.T = 8
        self.hermitian = False
        self.residual = True        
        self.conv_layers.append(DirectedUnitaryGCNConvLayer(self.input_dim, self.hidden_dim, dropout = dropout, residual = False))
        for _ in range(self.num_layers):
            self.conv_layers.append(DirectedUnitaryGCNConvLayer(self.hidden_dim, self.hidden_dim, residual = self.residual, dropout = dropout))
        
        for _ in range(self.num_layers + 1): # new
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.GELU(),
                # nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.GELU(),
                # nn.Linear(self.hidden_dim, self.hidden_dim),
                # dropout
                # nn.Dropout(p=dropout)
            ))
        
        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x_in, edge_index):
        for i, layer in enumerate(self.conv_layers):
            x_in = layer(x_in, edge_index)
            x_mlp = self.mlp_layers[i](x_in)
            x_in = F.normalize(x_mlp + x_in, p=2, dim=1)
            x_in = x_in + x_mlp
            
        return torch.nn.functional.log_softmax(self.final_mlp(x_in.real), dim=1)


class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        weight_decay,
        train_mask,
        val_mask,
        test_mask,
        evaluator=None,
        scheduler_t_max: int = None,
        scheduler_eta_min: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

        # scheduler hyperparameters
        self.scheduler_t_max = scheduler_t_max
        self.scheduler_eta_min = scheduler_eta_min

        # save_hyperparameters will log lr, weight_decay, scheduler_* etc.
        self.save_hyperparameters(
            ignore=['model', 'train_mask', 'val_mask', 'test_mask', 'evaluator']
        )

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        val_acc   = self.evaluate(y_pred=y_pred[self.val_mask],   y_true=y[self.val_mask])

        self.log("train_loss", loss,     on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc",  train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",    val_acc,   on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            return self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            return y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)
        y_pred = out.max(1)[1]
        test_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            # self.model.parameters(),
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # length of each cosine cycle = half of total epochs
        total_epochs = 1000 # self.scheduler_t_max or self.trainer.max_epochs
        half_epochs  = total_epochs // 2

        # two identical cosine-annealing cycles
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=half_epochs,      # epochs per cycle before restart
            T_mult=1,              # keep cycle length constant
            eta_min=self.scheduler_eta_min
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',    # step() at end of each epoch
                'frequency': 1,
                'name': 'cosine_restart_lr'
            }
        }
    
    
def get_model(args):
    if args.model == "gnn":
        return GNN(
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            conv_type=args.conv_type,
            jumping_knowledge=args.jk,
            normalize=args.normalize,
            alpha=args.alpha,
            learn_alpha=args.learn_alpha,
        )
    elif args.model == "unigcn":
        return UnitaryGCN(
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
        )
    elif args.model == "dune":
        return DirectedUnitaryGCN(
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
        )
