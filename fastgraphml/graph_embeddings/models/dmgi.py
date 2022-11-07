import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from arango.database import Database
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj, EdgeType, OptPairTensor

from ..utils import GraphUtils

# check for gpu
device = "cuda" if torch.cuda.is_available() else "cpu"


class DMGI(torch.nn.Module):
    """Deep Multiplex Graph Infomax (DMGI) model
    modified from '<https://github.com/pyg-team/pytorch_geometric/blob/6267de93c6b04f46a306aa58e414de330ef9bb10/examples/hetero/dmgi_unsup.py>'.  # noqa: E501

    :database (type: Database): A python-arango database instance.
    :arango_graph (type: str): The name of ArangoDB graph which we want to export to PyG
    :metagraph (type: dict): It exports ArangoDB graphs to PyG data objects. We define
        metagraph as a dictionary defining vertex & edge collections to import to PyG,
        along with collection-level specifications to indicate which ArangoDB attributes
        will become PyG features/labels. It also supports different encoders such as
        identity and categorical encoder on database attributes. Detailed information
        regarding different use cases and metagraph definitons can be found on
        adbpyg_adapter github page i.e <https://github.com/arangoml/pyg-adapter>.
    :metapaths (type: list(list[Tuple(str,str,str))]): It is described as list of list
        of tuples. Adds additional edge types to a Hetero Graph between the source node
        type and the destination node type of a given metapath.
        The metapath defined as (src_node_type, rel_type, dst_node_type) tuples.
    e.g. Adding two metapaths (reference: <https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.AddMetaPaths>):# noqa: E501
    # 1. From "paper" to "paper" through "conference"
    # 2. From "author" to "conference" through "paper"
    metapaths = [[("paper", "conference"), ("conference", "paper")],
         [("author", "paper"), ("paper", "conference")]]
    :key_node (type: str): Node type on which we want to test the performance of
        generated graph embeddings. Performance is tested using node classification task
    :pyg_graph (type: PyG data object): It generates graph embeddings using PyG graphs
        (via PyG data objects) directy rather than ArangoDB graphs. When generating
        graph embeddings via PyG graphs, database=arango_graph=metagraph=None.
    :embedding_size (type: int): Length of the node embeddings when they are mapped to
        d-dimensional euclidean space.
    :dropout_perc (type: float): Handles overfitting inside the model.
    :transform (type: torch_geometric.transforms): It is used to transform PyG data
        objects. Various transformation methods can be chained together using Compose.
                for e.g. transform = T.Compose([
                        T.NormalizeFeatures(),
                        T.RandomNodeSplit(num_val=0.2, num_test=0.1)])
    :num_val (type: float): Percentage of nodes selected for validation set.
    :num_test (type: float): Percentage of nodes selected for test set.

    Note: After selecting the percentage for validation and test nodes,
    rest percentage of the nodes are considered as training nodes.
    """

    def __init__(
        self,
        database: Database = None,
        arango_graph: Optional[str] = None,
        metagraph: Union[Dict[str, object], None] = None,
        metapaths: Optional[List[EdgeType]] = None,
        key_node: Optional[str] = None,
        pyg_graph: Data = None,
        embedding_size: int = 64,
        dropout_perc: float = 0.5,
        transform: Optional[List[Callable[..., Any]]] = None,
        num_val: float = 0.1,
        num_test: float = 0.1,
    ):
        super().__init__()

        if (
            database is not None or arango_graph is not None or metagraph is not None
        ) and pyg_graph is not None:
            msg = "when generating graph embeddings via PyG data objects, database=arango_graph=metagraph=None and vice versa"  # noqa: E501
            raise Exception(msg)

        if database is not None:
            if not issubclass(type(database), Database):
                msg = "**db** parameter must inherit from arango.database.Database"
                raise TypeError(msg)

        # arango to PyG
        self.graph_util = GraphUtils(
            arango_graph,
            metagraph,
            database,
            pyg_graph,
            num_val,
            num_test,
            transform,
            key_node,
            metapaths,
        )
        # get PyG graph
        G = self.graph_util.graph
        G = G.to(device)
        self.G = G
        self.x = G[key_node].x.float()
        self.edge_indices = G.edge_index_dict.values()
        self.key_node = key_node

        self.metadata = G.metadata()
        self.dropout_perc = dropout_perc

        num_nodes = G[key_node].num_nodes
        in_channels = G[key_node].x.size(-1)
        out_channels = embedding_size
        num_relations = len(G.edge_types)
        # relation type specific node encoder
        # generates realtion type specific node embedding
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)]
        )
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        self.Z = torch.nn.Parameter(torch.Tensor(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_indices: Adj
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=self.dropout_perc, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=self.dropout_perc, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))

        return pos_hs, neg_hs, summaries

    def loss(
        self, pos_hs: List[Tensor], neg_hs: List[Tensor], summaries: List[Tensor]
    ) -> Tensor:
        loss = 0.0
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss += 0.001 * (pos_reg_loss - neg_reg_loss)

        return loss

    # save checkpoints whenever there is an increase in validation accuracy
    @staticmethod
    def save_checkpoints(
        state: Dict[str, str], is_best: bool, ckp_path: str, best_model_path: str
    ) -> None:
        file_path = ckp_path
        torch.save(state, file_path)
        # if it is a best model, min train loss
        if is_best:
            best_file_path = best_model_path
            # copy best checkpoint file to best model path
            shutil.copyfile(file_path, best_file_path)

    def _train(
        self,
        ckp_path: str = "./latest_model_checkpoint.pt",
        best_model_path: str = "./best_model.pt",
        epochs: int = 51,
        lr: float = 0.0005,
        **kwargs: Any,
    ) -> None:
        """Train GraphML model.

        :ckp_path (type: str): Path to save model's latest checkpoints
            (i.e. at each epoch). Pytorch models are saved with .pt file extension.
            By default it saves model in cwd.
        :best_model_path (type: str): Path to save model whenever there is an increase
            in validation accuracy. By default it saves model in cwd.
        :epochs (type: int): Number of times training data go through the model.
        :lr (type: float): Learning rate.
        :**kwargs: Additional arguments for the Adam optimizer for
            e.g. weight_decay, betas, etc.
        """

        model = self.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr, **kwargs)
        best_acc = 0.0
        print("Training started .........")
        for epoch in range(1, epochs):
            model.train()
            optimizer.zero_grad()
            pos_hs, neg_hs, summaries = model(self.x, self.edge_indices)
            loss = model.loss(pos_hs, neg_hs, summaries)
            loss.backward()
            loss = float(loss)
            optimizer.step()

            ##################
            # validate model #
            ##################

            val_acc, test_acc = self.val()

            print(
                f"Epoch: {epoch:03d}, Train_Loss: {loss:.4f}, "
                f"Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

            # create checkpoint variable and add crucial model information
            model_checkpoint = {
                "epoch": epoch,
                "best_acc": val_acc,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # save current/last checkpoint
            # used for resuming training
            self.save_checkpoints(model_checkpoint, False, ckp_path, best_model_path)

            # save model if val acc increases
            if val_acc > best_acc:
                print(
                    "Val Acc increased ({:.5f} --> {:.5f}).  Saving model ...".format(
                        best_acc, val_acc
                    )
                )
                # save checkpoint as best model
                self.save_checkpoints(model_checkpoint, True, ckp_path, best_model_path)
                best_acc = val_acc

    @torch.no_grad()
    def val(
        self,
    ) -> Tuple[float, float]:
        """Tests the performance of a generated graph embeddings using Node
        Classification as a downstream task.

        returns validation and test accuracy.

        model: Graph embedding model.
        """
        self.eval()
        train_emb = self.Z[self.G[self.key_node].train_mask].cpu()
        val_emb = self.Z[self.G[self.key_node].val_mask].cpu()
        test_emb = self.Z[self.G[self.key_node].test_mask].cpu()

        train_y = self.G[self.key_node].y[self.G[self.key_node].train_mask].cpu()
        val_y = self.G[self.key_node].y[self.G[self.key_node].val_mask].cpu()
        test_y = self.G[self.key_node].y[self.G[self.key_node].test_mask].cpu()
        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        clf.fit(train_emb, train_y)
        val_acc = clf.score(val_emb, val_y)
        test_acc = clf.score(test_emb, test_y)

        return val_acc, test_acc

    @torch.no_grad()
    def get_embeddings(
        self,
    ) -> Dict[Optional[str], Any]:
        """Returns Graph Embeddings for the key node only.

        Embeddings size: (n, embedding_size), where
        n: number of nodes present for key node inside graph.
        embedding_size: Length of the node embeddings when they are mapped to
            d-dimensional euclidean space.

        """
        emb = {}
        self.eval()
        z = self.Z.detach().cpu().numpy()
        emb[self.key_node] = z

        return emb
