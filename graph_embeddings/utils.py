# utils file
import math

import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import umap
from adbpyg_adapter import ADBPyG_Adapter
from arango import ArangoClient
from rich.progress import track
from torch_geometric.data import Data
from tqdm import tqdm


# various Graph ML utilites
class GraphUtils(nn.Module):
    """Various graph utility hepler methods"""

    def __init__(
        self,
        arango_graph,
        metagraph,
        database,
        pyg_graph,
        num_val,
        num_test,
        key_node=None,
        metapaths=None,
    ):
        super().__init__()
        self.num_val = num_val
        self.num_test = num_test
        self.key_node = key_node
        self.metapaths = metapaths
        self.database = database
        self.metagraph = metagraph

        if pyg_graph == None:
            self.graph = self.arango_to_pyg(arango_graph, metagraph)
            print(self.graph_stats())
        else:
            self.graph = self.pyg_preprocess(pyg_graph)
            print(self.graph_stats())

    # Takes PyG data object and preprocess it
    def pyg_preprocess(self, pyg_data):
        if self.key_node != None and self.metapaths != None:
            if hasattr(pyg_data[self.key_node], "train_mask"):
                pyg_data = pyg_data
                pyg_data = T.AddMetaPaths(
                    self.metapaths, drop_orig_edges=True, drop_unconnected_nodes=True
                )(pyg_data)
            else:
                pyg_data = T.RandomNodeSplit(
                    num_val=self.num_val, num_test=self.num_test
                )(pyg_data)
                pyg_data = T.AddMetaPaths(
                    self.metapaths, drop_orig_edges=True, drop_unconnected_nodes=True
                )(pyg_data)

        elif self.key_node != None and self.metapaths == None:
            if hasattr(pyg_data[self.key_node], "train_mask"):
                pyg_data = pyg_data
            else:
                pyg_data = T.RandomNodeSplit(
                    num_val=self.num_val, num_test=self.num_test
                )(pyg_data)

        else:
            if pyg_data.has_isolated_nodes():
                print("Removing isolated nodes.....")
                pyg_data = T.RemoveIsolatedNodes()(pyg_data)
            else:
                pyg_data = pyg_data

            if hasattr(pyg_data, "train_mask"):
                pyg_data = pyg_data
            else:
                pyg_data = T.RandomNodeSplit(
                    num_val=self.num_val, num_test=self.num_test
                )(pyg_data)

        return pyg_data

    # export ArangoDB graph to Pyg data object
    def arango_to_pyg(self, arango_graph, metagraph):
        adbpyg = ADBPyG_Adapter(self.database)
        pyg_obj = adbpyg.arangodb_to_pyg(arango_graph, metagraph)
        pyg_obj = self.pyg_preprocess(pyg_obj)
        return pyg_obj

    # print graph statistics information about input graph for e.g. num of nodes, edges, etc.
    def graph_stats(
        self,
    ):
        if self.key_node == None:
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
                if total_nodes == None:
                    # counting documents in ArangoDB collections and adding it to num_nodes attribute
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

    def visualize_embeddings(
        self, graph_emb, class_mapping=None, node_type=None, emb_percent=0.1
    ):
        """Performs dimensionality reduction (2D) and visualization of graph embeddings using U-Map

        :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
        n: number of nodes in graph
        embedding_size: Length of the node embeddings when they are mapped to d-dimensional euclidean space.
        :class_mapping (type: dict): It is a dictionary where class names are mapped to integer labels.
        If class_mappings are not provided the method uses integers labels as legend inside the figure. for e.g.
        {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',4: 'Computer Components',
        5: 'Video Projectors',6: 'Routers',7: 'Tablets',8: 'Networking Products',9: 'Webcams'} # amazon computer dataset.
        :node_type (type: str):  Node type for which we want to retreive embeddings to perform visualization. Used to store Hetero graph embeddings.
        :emb_perc (type: float): Percentage of embeddings to visualize, 0.1 means 10% of data will be visualized.
        """

        if node_type != None:
            y_np = self.graph[node_type].y.cpu().numpy()
        else:
            y_np = self.graph.y.cpu().numpy()

        num_nodes = math.floor(graph_emb.shape[0] * emb_percent)
        num_y = math.floor(y_np.shape[0] * emb_percent)
        graph_emb = graph_emb[:num_nodes]
        labels = y_np[:num_y]
        umap_embd = umap.UMAP().fit_transform(graph_emb)
        plt.figure(figsize=(8, 8))
        if class_mapping != None:
            palette = {}
            class_names = [class_mapping[l] for l in labels]
            for n, y in enumerate(set(np.array(class_names))):
                palette[y] = f"C{n}"
            sns.scatterplot(
                x=umap_embd.T[0],
                y=umap_embd.T[1],
                hue=np.array(class_names),
                palette=palette,
            )
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            plt.savefig("./umap_embd_visualization.png", dpi=120)
        else:
            palette = {}
            for n, y in enumerate(set(labels)):
                palette[y] = f"C{n}"
            sns.scatterplot(
                x=umap_embd.T[0], y=umap_embd.T[1], hue=labels, palette=palette
            )
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
            plt.savefig("./umap_embd_visualization.png", dpi=120)

    def similarity_search(
        self, graph_emb, top_k_nbors=10, nlist=10, search_type="exact"
    ):
        """Performs similarity search in sets of vectors of any size, up to ones that possibly do not fit in RAM using FAISS
        <https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/>.

        returns dist, nbors
            nbors: node ids with similar embeddings
            dist: corresponding similarity scores

        :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
        n: number of nodes in graph
        embedding_size: Length of the node embeddings when they are mapped to d-dimensional euclidean space.
        :top_k_nbors (type: int): Returns top-k nearest neighbors of all embeddings.
        :nlist (type: int): Number of clusters to be generated.
        :search_type (type: str): We support two types of search for now:
        1. exact search: For precise similarity search but at the cost of scalability.
        2. approx search: For scalable similarity search but at the cost of some precision loss.
        """

        emb_dim = graph_emb.shape[1]
        # exact search
        if search_type == "exact":
            # inner product search index
            search_index = faiss.IndexFlatIP(emb_dim)
            # normalize the embeddings and add to search index
            faiss.normalize_L2(graph_emb)
            search_index.add(graph_emb)
            # search top-k nearest neighbors of all embeddings
            dist, nbors = search_index.search(graph_emb, k=top_k_nbors + 1)

        # approx search
        elif search_type == "approx":
            # assign embeddings to specific cluster using Inner product
            quantiser = faiss.IndexFlatIP(emb_dim)
            # L2 normalization of embeddings
            faiss.normalize_L2(graph_emb)
            # creates partitioned search index
            search_index = faiss.IndexIVFFlat(
                quantiser, emb_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            # train and add the embeddings to the index
            search_index.train(graph_emb)
            search_index.add(graph_emb)
            # nbors: top k node ids with highest cosine sim
            # dist: cosine distances of top k nodes
            dist, nbors = search_index.search(graph_emb, k=top_k_nbors + 1)

        else:
            assert (
                search_type == "exact" or search_type == "approx"
            ), "pass search type either exact or approx"

        return dist, nbors

    # store graph embeddings inside arangodb
    def store_embeddings(
        self,
        graph_emb,
        collection_name=None,
        batch_size=100,
        class_mapping=None,
        node_type=None,
        nearest_nbors_search=False,
        top_k_nbors=10,
        nlist=10,
        search_type="exact",
    ):
        """Store generated graph embeddings inside ArangoDB.

        :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
        n: number of nodes in graph
        embedding_size: Length of the node embeddings when they are mapped to d-dimensional euclidean space.
        :collection_name (type: str): Name of the document collection where we want to store graph embeddings.
        :batch_size (type: int): Batch size.
        :class_mapping (type: dict): It is a dictionary where class names are mapped to integer labels.
        If class_mappings are not provided the method uses integers labels as legend inside the figure. for e.g.
        {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',4: 'Computer Components',
        5: 'Video Projectors',6: 'Routers',7: 'Tablets',8: 'Networking Products',9: 'Webcams'} # amazon computer dataset.
        :node_type (type: str):  Node type for which we want to store embeddings. Used to store Hetero graph embeddings.
        :nearest_nbors_search (type: bool): If nearest_nbors_search=True, store_embeddings method saves generated Graph Embeddings in ArangoDB
        along with top_k nearest neighbors (node ids with similar embeddings) and their corresponding similarity scores (i.e. cosine distance).
        :top_k_nbors (type: int): Returns top-k nearest neighbors of all embeddings.
        :nlist (type: int): Number of clusters to be generated.
        :search_type (type: str): We support two types of search for now:
        1. exact search: For precise similarity search but at the cost of scalability.
        2. approx search: For scalable similarity search but at the cost of some precision loss.
        """

        assert (
            collection_name != None
        ), "pass arangodb collection name to store embeddings "

        # create document collection with name "collection_name" in arangodb
        if not self.database.has_collection(collection_name):
            self.database.create_collection(collection_name, replication_factor=3)

        batch = []
        batch_idx = 1
        index = 0
        emb_collection = self.database[collection_name]

        if nearest_nbors_search == True:
            dist, nbors = self.similarity_search(graph_emb)

        for idx in track(range(graph_emb.shape[0])):
            insert_doc = {}
            insert_doc["_id"] = collection_name + "/" + str(idx)
            insert_doc["embedding"] = graph_emb[idx].tolist()
            ## add class names
            if class_mapping != None and self.key_node == None:
                insert_doc["label"] = self.graph.y[idx].item()
                insert_doc["class_name"] = class_mapping[self.graph.y[idx].item()]

            elif class_mapping != None and self.key_node != None:
                if hasattr(self.graph[node_type], "y"):
                    insert_doc["label"] = self.graph[node_type].y[idx].item()
                    insert_doc["class_name"] = class_mapping[
                        self.graph[node_type].y[idx].item()
                    ]
                else:
                    pass

            elif class_mapping == None and self.key_node != None:
                if hasattr(self.graph[node_type], "y"):
                    insert_doc["label"] = self.graph[self.key_node].y[idx].item()
                else:
                    pass

            elif class_mapping == None and self.key_node == None:
                insert_doc["label"] = self.graph.y[idx].item()
            else:
                pass
            ## add top-k nearest neighbours
            if nearest_nbors_search == True:
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
