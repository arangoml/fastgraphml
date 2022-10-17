import torch
import torch.nn as nn
from typing import Any, List, Tuple
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from torch.nn import Linear as Lin
from torch import Tensor
import shutil
from arango.database import Database
import numpy as np
from ..utils import GraphUtils


# check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# neighborhood sampling
class NeighborSampler(RawNeighborSampler):
    """ For each node in batch, it sample a direct neighbor (as positive
        example) and a random node (as negative example):

        returns sampled neighborhood
    """

    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]
        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                dtype=torch.long)
        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)



class GAT(torch.nn.Module):
    """ GATCONV model modified from '<https://github.com/pyg-team/pytorch_geometric/blob/6267de93c6b04f46a306aa58e414de330ef9bb10/examples/gat.py>'
    
    :database (type: Database): A python-arango database instance.
    :arango_graph (type: str): The name of ArangoDB graph which we want to export to PyG.
    :metagraph (type: dict): It exports ArangoDB graphs to PyG data objects. We define metagraph as 
                a dictionary defining vertex & edge collections to import to PyG, along 
                with collection-level specifications to indicate which ArangoDB attributes will become PyG features/labels. 
                It also supports different encoders such as identity and categorical encoder on database attributes. Detailed information regarding different
                use cases and metagraph definitons can be found on adbpyg_adapter github page i.e <https://github.com/arangoml/pyg-adapter>.
    :pyg_graph (type: PyG data object): It generates graph embeddings using PyG graphs (via PyG data objects) directy rather than ArangoDB graphs.
                When generating graph embeddings via PyG graphs, database=arango_graph=metagraph=None.
    :embedding_size (type: int): Length of the node embeddings when they are mapped to d-dimensional euclidean space.
    :heads (type: int): Number of attention heads. Model learns to give attention to only important nodes in node's neighborhood.
    :num_layers (type: int): Number of GraphSage Layers.
    :sizes (type: [int, int]): Number of neighbors to select at each layer for every node (uniform random sampling) in order to perform neighborhood sampling. 
    :batch_size (type: int): Number of nodes to be present inside batch along with their neighborhood. Used while performing neighborhood sampling.
    :dropout_perc (type: float): Handles overfitting inside the model
    :shuffle (type: bool): If set to True, it shuffles data before performing neighborhood sampling.
    :num_val (type: float): Percentage of nodes selected for validation set.
    :num_test (type: float): Percentage of nodes selected for test set.
    
    Note: After selecting the percentage for validation and test nodes, rest percentage of the nodes are considered as training nodes.

    :**kwargs: Additional arguments of the GATConv class. For more arguments please refer the following link
     <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv>
    """

    def __init__(self, database = None, arango_graph = None, metagraph = None, pyg_graph = None, embedding_size = 64, heads = 2,
        num_layers = 2, sizes = [10, 10], batch_size = 256, dropout_perc = 0.5,  shuffle = True, num_val = 0.1, num_test = 0.1, **kwargs):
        super().__init__()

        if (database is not None or arango_graph is not None or metagraph is not None) and pyg_graph is not None:
            msg = "when generating graph embeddings via PyG data objects, database=arango_graph=metagraph=None and vice versa"
            raise Exception(msg)

        if database is not None:
            if not issubclass(type(database), Database):
                msg = "**db** parameter must inherit from arango.database.Database"
                raise TypeError(msg)

        # arango to Pyg
        self.graph_util = GraphUtils(arango_graph, metagraph, database, pyg_graph, num_val, num_test)
        # get PyG graph
        G = self.graph_util.graph
        self.in_channels = G.num_node_features
        self.hidden_channels = embedding_size
        self.G = G
        self.x = G.x.float().to(device)
        self.edge_index = G.edge_index.to(device)
        self.num_nodes = G.num_nodes
        self.num_layers = num_layers
        self.sizes = sizes
        self.batch_size = batch_size 
        self.dropout_perc = dropout_perc
        self.shuffle = shuffle

        # create train loader 
        self.train_loader = NeighborSampler(self.edge_index, batch_size=self.batch_size, sizes=self.sizes,
                                            shuffle=self.shuffle, num_nodes=self.num_nodes)


        # defining GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(self.in_channels, self.hidden_channels, heads))
            else:
                self.convs.append(GATConv(heads * self.hidden_channels, self.hidden_channels, heads))
            
        # adding skip connections
        self.skips = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.skips.append(Lin(self.in_channels, self.hidden_channels * heads))
            else:
                self.skips.append(Lin(self.hidden_channels * heads, self.hidden_channels * heads))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_perc, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) + self.skips[i](x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_perc, training=self.training)
        return x

    # save checkpoints whenever there is an increase in validation accuracy
    @staticmethod
    def save_checkpoints(state, is_best, ckp_path, best_model_path):
        file_path = ckp_path
        torch.save(state, file_path)
        #if it is a best model, min train loss
        if is_best:
            best_file_path = best_model_path
            # copy best checkpoint file to best model path 
            shutil.copyfile(file_path, best_file_path)
    

    def _train(self, model, ckp_path = "./latest_model_checkpoint.pt", best_model_path = "./best_model.pt", epochs = 51, lr = 0.001, **kwargs):
        """Train GraphML model.

        :model: Graph embedding model.
        :ckp_path (type: str): Path to save model's latest checkpoints (i.e. at each epoch). Pytorch models are saved with .pt file extension.
         By default it saves model in cwd.
        :best_model_path (type: str): Path to save model whenever there is an increase in validation accuracy. By default it saves model in cwd.
        :epochs (type: int): Number of times training data go through the model.
        :lr (type: float): Learning rate.
        :**kwargs: Additional arguments for the Adam optimizer for e.g. weight_decay, betas, etc.
        """

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr, **kwargs)
        best_acc = 0.0
        print('Training started .........')
        for epoch in range(1, epochs):
            model.train()
            total_loss = 0
            for batch_size, n_id, adjs in self.train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()

                out = model(self.x[n_id], adjs)
                out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

                pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
                neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
                loss = -pos_loss - neg_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss) * out.size(0)
            # print unsupervised GraphSage loss
            train_loss = total_loss / self.num_nodes

            ##################
            # validate model #
            ##################

            val_acc, test_acc = self.val(model)

            print(f'Epoch: {epoch:03d}, Train_Loss: {loss:.4f}, 'f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            # create checkpoint variable and add crucial model information
            model_checkpoint = {
            'epoch': epoch,
            'best_acc': val_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}

            # save current/last checkpoint
            # used for resuming training
            self.save_checkpoints(model_checkpoint, False, ckp_path, best_model_path)

            # save model if val acc increases
            if val_acc > best_acc:
                print('Val Acc increased ({:.5f} --> {:.5f}).  Saving model ...'.format(best_acc, val_acc))
                # save checkpoint as best model
                self.save_checkpoints(model_checkpoint, True, ckp_path, best_model_path)
                best_acc = val_acc

    
    @torch.no_grad()
    def val(self, model):
        """Tests the performance of a generated graph embeddings using Node Classification as a downstream task.
        
        returns validation and test accuracy.

        model: Graph embedding model.
        """
        model.eval()
        out = model.full_forward(self.x, self.edge_index).cpu()
        clf = LogisticRegression(max_iter=400, class_weight='balanced')
        clf.fit(out[self.G.train_mask], self.G.y[self.G.train_mask])
        val_acc = clf.score(out[self.G.val_mask], self.G.y[self.G.val_mask])
        test_acc = clf.score(out[self.G.test_mask], self.G.y[self.G.test_mask])

        return val_acc, test_acc

    @torch.no_grad()
    def get_embeddings(self, model):
        """Returns Graph Embeddings of size (n, embedding_size),
           n: number of nodes in graph.
           embedding_size: Length of the node embeddings when they are mapped to d-dimensional euclidean space.

        model: Graph embedding model.
        """
        model.eval()
        emb = model.full_forward(self.x, self.edge_index).detach().cpu().numpy()
        return emb