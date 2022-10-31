import math
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import umap
from torch_geometric.data import Data


def visualize_embeddings(
    graph: Data,
    graph_emb: Union[npt.NDArray[np.float64], Any],
    class_mapping: Optional[Dict[int, str]] = None,
    node_type: Optional[str] = None,
    emb_percent: float = 0.1,
) -> None:
    """Performs dimensionality reduction (2D) and visualization of graph embeddings
    using U-Map.

    :graph (type: PyG data object): PyG data object which can be accessed using model.G
        inside the framework.
    :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
    n: number of nodes in graph,
    embedding_size: Length of the node embeddings when they are mapped to d-dimensional
        euclidean space.
    :class_mapping (type: dict): It is a dictionary where class names are mapped to
        integer labels. If class_mappings are not provided the method uses integers
        labels as legend inside the figure. for e.g.
        {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',
        4: 'Computer Components', 5: 'Video Projectors',6: 'Routers',7: 'Tablets',
        8: 'Networking Products',9: 'Webcams'} # amazon computer dataset.
    :node_type (type: str):  Node type for which we want to retreive embeddings to
         perform visualization. Used to store Hetero graph embeddings.
    :emb_perc (type: float): Percentage of embeddings to visualize, 0.1 means 10% of
         data will be visualized.
    """

    if node_type is not None:
        y_np = graph[node_type].y.cpu().numpy()
    else:
        y_np = graph.y.cpu().numpy()

    num_nodes = math.floor(graph_emb.shape[0] * emb_percent)
    num_y = math.floor(y_np.shape[0] * emb_percent)
    graph_emb = graph_emb[:num_nodes]
    labels = y_np[:num_y]
    umap_embd = umap.UMAP().fit_transform(graph_emb)
    plt.figure(figsize=(8, 8))
    if class_mapping is not None:
        palette = {}
        class_names = [class_mapping[label] for label in labels]
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
        sns.scatterplot(x=umap_embd.T[0], y=umap_embd.T[1], hue=labels, palette=palette)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.savefig("./umap_embd_visualization.png", dpi=120)
