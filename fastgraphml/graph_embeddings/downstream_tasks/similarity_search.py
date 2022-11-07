from typing import Any, Tuple, Union

import faiss
import numpy as np
import numpy.typing as npt


def similarity_search(
    graph_emb: Union[npt.NDArray[np.float64], Any],
    top_k_nbors: int = 10,
    nlist: int = 10,
    search_type: str = "exact",
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Performs similarity search in sets of vectors of any size, up to ones that
    possibly do not fit in RAM using FAISS.

    <https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/>.  # noqa: E501

    returns dist, nbors
        nbors: node ids with similar embeddings
        dist: corresponding similarity scores

    :graph_emb (type: 2D numpy array): Numpy array of size (n, embedding_size),
    n: number of nodes in graph
    embedding_size: Length of the node embeddings when they are mapped to
        d-dimensional euclidean space.
    :top_k_nbors (type: int): Returns top-k nearest neighbors of all embeddings.
    :nlist (type: int): Number of clusters to be generated.
    :search_type (type: str): We support two types of search for now:
    1. exact search: For precise similarity search but at the cost of scalability.
    2. approx search: For scalable similarity search but at the cost of some precision
        loss.
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
        raise Exception("Pass search type either exact or approx.")

    return dist, nbors
