# utils file
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch_geometric.transforms as T
from adbpyg_adapter import ADBPyG_Adapter
from arango.database import Database
from arango.exceptions import CollectionCreateError
from rich.progress import track
from torch_geometric.data import Data
from torch_geometric.typing import EdgeType

from .downstream_tasks.similarity_search import similarity_search


# various Graph ML utilites
class GraphUtils(nn.Module):
    """Various graph utility hepler methods."""

    def __init__(
        self,
        arango_graph: Optional[str],
        metagraph: Union[Dict[str, object], None],
        database: Database,
        pyg_graph: Data,
        num_val: float,
        num_test: float,
        transform: Any,
        key_node: Union[str, None] = None,
        metapaths: Optional[List[EdgeType]] = None,
    ):
        super().__init__()
        self.num_val = num_val
        self.num_test = num_test
        self.key_node = key_node
        self.metapaths = metapaths
        self.database = database
        self.metagraph = metagraph
        self.transform = transform

        if pyg_graph is None:
            self.graph = self.arango_to_pyg(arango_graph, metagraph)
            print(self.graph_stats())
        else:
            self.graph = self.pyg_preprocess(pyg_graph)
            print(self.graph_stats())

    def pyg_preprocess(self, pyg_data: Data) -> Data:
        """Takes PyG data object and preprocess it.
        By default it performs following preporocessing:
        1. AddMetaPaths
        2. Perform Random Node Split i.e. splitting data into train, val and test set.

        :pyg_data (tpye: PyG data object): PyG data object
        """
        # hetero graph
        if self.key_node is not None and self.metapaths is not None:
            if hasattr(pyg_data[self.key_node], "train_mask"):
                pyg_data = pyg_data
                pyg_data = T.AddMetaPaths(
                    self.metapaths, drop_orig_edges=True, drop_unconnected_nodes=True
                )(pyg_data)
                if self.transform is not None:
                    pyg_data = self.transform(pyg_data)
            else:
                pyg_data = T.RandomNodeSplit(
                    num_val=self.num_val, num_test=self.num_test
                )(pyg_data)
                pyg_data = T.AddMetaPaths(
                    self.metapaths, drop_orig_edges=True, drop_unconnected_nodes=True
                )(pyg_data)
                if self.transform is not None:
                    pyg_data = self.transform(pyg_data)

        elif self.key_node is not None and self.metapaths is None:
            if hasattr(pyg_data[self.key_node], "train_mask"):
                pyg_data = pyg_data
                if self.transform is not None:
                    pyg_data = self.transform(pyg_data)
            else:
                pyg_data = T.RandomNodeSplit(
                    num_val=self.num_val, num_test=self.num_test
                )(pyg_data)
                if self.transform is not None:
                    pyg_data = self.transform(pyg_data)
        # homo graph
        else:
            if hasattr(pyg_data, "train_mask"):
                pyg_data = pyg_data
                if self.transform is not None:
                    pyg_data = self.transform(pyg_data)
            else:
                pyg_data = T.RandomNodeSplit(
                    num_val=self.num_val, num_test=self.num_test
                )(pyg_data)
                if self.transform is not None:
                    pyg_data = self.transform(pyg_data)

        return pyg_data

    def arango_to_pyg(
        self, arango_graph: Optional[str], metagraph: Union[Dict[str, object], None]
    ) -> Data:
        """Exports ArangoDB graph to PyG data object using ArangoDB PyG Adapter.

        :arango_graph (type: str): The name of ArangoDB graph which we want to
        export to PyG.
        :metagraph (type: dict): It exports ArangoDB graphs to PyG data objects.
            We define metagraph as a dictionary defining vertex & edge collections
            to import to PyG, along with collection-level specifications to
            indicate which ArangoDB attributes will become PyG features/labels.
            It also supports different encoders such as identity and categorical
            encoder on database attributes. Detailed information regarding different
            use cases and metagraph definitons can be found on adbpyg_adapter github
            page i.e <https://github.com/arangoml/pyg-adapter>.
        """

        adbpyg = ADBPyG_Adapter(self.database)
        pyg_obj = adbpyg.arangodb_to_pyg(arango_graph, metagraph)
        pyg_obj = self.pyg_preprocess(pyg_obj)
        return pyg_obj

    # print graph statistics information about input graph for
    # e.g. num of nodes, edges, etc.
    def graph_stats(
        self,
    ) -> Dict[str, Union[str, int, float]]:
        if self.key_node is None:
            print("Homogeneous Graph Detected ........ \n")
            g_info = {}
            g_info["Nodes"] = self.graph.num_nodes
            g_info["Node_feature_size"] = self.graph.num_node_features
            g_info["Number_of_classes"] = self.graph.y.unique().size(0)
            g_info["Edges"] = self.graph.num_edges
            if "edge_attr" in self.graph:
                e_size = self.graph.edge_attr.size(1)
                g_info["Edge_feature_fize"] = e_size
            else:
                g_info["Edge_feature_fize"] = None
            if self.graph.is_directed():
                g_info["Graph Directionality"] = "Directed"
            else:
                g_info["Graph Directionality"] = "Undirected"
            g_info[
                "Average node degree"
            ] = f"{self.graph.num_edges/ self.graph.num_nodes:.2f}"
            g_info["Number of training nodes"] = self.graph.train_mask.sum().item()
            g_info["Number of val nodes"] = self.graph.val_mask.sum().item()
            g_info["Number of test nodes"] = self.graph.test_mask.sum().item()
            g_info["Has isolated nodes"] = self.graph.has_isolated_nodes()
            return g_info
        else:
            print("Heterogeneous Graph Detected .......... \n")
            g_info = {}
            node_types = self.graph.metadata()[0]
            edge_types = self.graph.metadata()[1]

            # adding node information
            node_data = {}
            for node in node_types:
                total_nodes = 0
                total_nodes = self.graph[node].num_nodes
                if total_nodes is None:
                    # counting documents in ArangoDB collections and adding it
                    # to num_nodes attribute
                    total_nodes = self.database.collection(node).count()
                    self.graph[node].num_nodes = total_nodes
                if hasattr(self.graph[node], "train_mask"):
                    node_data["Number of " + node + " nodes"] = total_nodes
                    node_data["Number of train " + node + " nodes"] = (
                        self.graph[node].train_mask.sum().item()
                    )
                    node_data["Number of val " + node + " nodes"] = (
                        self.graph[node].val_mask.sum().item()
                    )
                    node_data["Number of test " + node + " nodes"] = (
                        self.graph[node].test_mask.sum().item()
                    )
                    node_data["Number of classes in " + node + " nodes"] = (
                        self.graph[node].y.unique().size(0)
                    )
                else:
                    node_data["number of " + node + " nodes"] = total_nodes

            # add total nodes
            g_info["Nodes"] = self.graph.num_nodes
            # add total edges
            g_info["Edges"] = self.graph.num_edges
            # add all node types
            g_info["node_types"] = node_types
            # add all edge_types
            g_info["edge_types"] = edge_types
            # check graph directionality
            if self.graph.is_directed():
                g_info["Graph Directionality"] = "Directed"
            else:
                g_info["Graph Directionality"] = "Undirected"
            # check for isolated nodes
            g_info["Has isolated nodes"] = self.graph.has_isolated_nodes()
            g_info["node stats"] = node_data
            return g_info

    # store graph embeddings inside arangodb
    def store_embeddings(
        self,
        graph_emb: Union[npt.NDArray[np.float64], Any],
        collection_name: str,
        batch_size: int = 100,
        class_mapping: Optional[Dict[int, str]] = None,
        node_type: Optional[str] = None,
        nearest_nbors_search: bool = False,
        top_k_nbors: int = 10,
        nlist: int = 10,
        search_type: str = "exact",
    ) -> None:
        """Store generated graph embeddings inside ArangoDB.

        :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
        n: number of nodes in graph
        embedding_size: Length of the node embeddings when they are mapped to
            d-dimensional euclidean space.
        :collection_name (type: str): Name of the document collection where we want to
            store graph embeddings.
        :batch_size (type: int): Batch size.
        :class_mapping (type: dict): It is a dictionary where class names are mapped
            to integer labels.
        If class_mappings are not provided the method uses integers labels as legend
        inside the figure. for e.g.
        {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',
        4: 'Computer Components', 5: 'Video Projectors',6: 'Routers',7: 'Tablets',
        8: 'Networking Products',9: 'Webcams'} # amazon computer dataset.
        :node_type (type: str):  Node type for which we want to store embeddings.
            Used to store Hetero graph embeddings.
        :nearest_nbors_search (type: bool): If nearest_nbors_search=True,
            store_embeddings method saves generated Graph Embeddings in ArangoDB
            along with top_k nearest neighbors (node ids with similar embeddings)
            and their corresponding similarity scores (i.e. cosine distance).
        :top_k_nbors (type: int): Returns top-k nearest neighbors of all embeddings.
        :nlist (type: int): Number of clusters to be generated.
        :search_type (type: str): We support two types of search for now:
        1. exact search: For precise similarity search but at the cost of scalability.
        2. approx search: For scalable similarity search but at the cost of some
            precision loss.
        """

        if collection_name is None:
            raise Exception("Pass arangodb collection name to store embeddings.")

        # create document collection with name "collection_name" in arangodb
        if not self.database.has_collection(collection_name):
            try:
                self.database.create_collection(collection_name, replication_factor=3)
            except CollectionCreateError as exec:
                raise CollectionCreateError(exec)

        batch = []
        batch_idx = 1
        index = 0
        emb_collection = self.database[collection_name]

        if nearest_nbors_search is True:
            dist, nbors = similarity_search(graph_emb)

        for idx in track(range(graph_emb.shape[0])):
            insert_doc = {}
            insert_doc["_id"] = collection_name + "/" + str(idx)
            insert_doc["embedding"] = graph_emb[idx].tolist()
            # add class names
            if class_mapping is not None and self.key_node is None:
                insert_doc["label"] = self.graph.y[idx].item()
                insert_doc["class_name"] = class_mapping[self.graph.y[idx].item()]

            elif class_mapping is not None and self.key_node is not None:
                if hasattr(self.graph[node_type], "y"):
                    insert_doc["label"] = self.graph[node_type].y[idx].item()
                    insert_doc["class_name"] = class_mapping[
                        self.graph[node_type].y[idx].item()
                    ]
                else:
                    pass

            elif class_mapping is None and self.key_node is not None:
                if hasattr(self.graph[node_type], "y"):
                    insert_doc["label"] = self.graph[self.key_node].y[idx].item()
                else:
                    pass

            elif class_mapping is None and self.key_node is None:
                insert_doc["label"] = self.graph.y[idx].item()
            else:
                pass
            # add top-k nearest neighbours
            if nearest_nbors_search is True:
                insert_doc["cosine_sim"] = dist[idx].tolist()
                insert_doc["similar_nodes"] = nbors[idx].tolist()
            batch.append(insert_doc)
            index += 1
            last_record = idx == (graph_emb.shape[0] - 1)
            if index % batch_size == 0:
                batch_idx += 1
                emb_collection.import_bulk(batch)
                batch = []
            if last_record and len(batch) > 0:
                print("Inserting batch the last batch!")
                emb_collection.import_bulk(batch)
