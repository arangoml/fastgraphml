import pytest
from torch_geometric.datasets import Planetoid

from fastgraphml.graph_embeddings import DMGI, GAT, METAPATH2VEC, SAGE, downstream_tasks

from .conftest import db  # type: ignore


def test_sage() -> None:
    metagraph = {
        "vertexCollections": {
            "cora_N": {"x": "x", "y": "y"},
        },
        "edgeCollections": {
            "cora_E": {},
        },
    }
    model = SAGE(db, "graph", metagraph, embedding_size=64)  # define model
    model._train(epochs=2)  # train
    embeddings = model.get_embeddings()  # get embeddings
    # check embeddings size
    assert embeddings.size == int(173312)
    # check similarity search
    dist, nbors = downstream_tasks.similarity_search(
        embeddings, top_k_nbors=10, nlist=10, search_type="exact"
    )
    assert nbors.size == int(29788)
    # check similarity search
    dist, nbors = downstream_tasks.similarity_search(
        embeddings, top_k_nbors=10, nlist=10, search_type="approx"
    )
    assert nbors.size == int(29788)
    # check similarity search
    with pytest.raises(Exception):
        dist, nbors = downstream_tasks.similarity_search(
            embeddings, top_k_nbors=10, nlist=10, search_type="wrong"
        )

    # check vis
    class_names = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g"}
    downstream_tasks.visualize_embeddings(
        model.G, embeddings, class_mapping=class_names
    )
    # check storing emb
    if db.has_collection("cora_emb"):
        db.delete_collection("cora_emb")
    model.graph_util.store_embeddings(
        embeddings,
        collection_name="cora_emb",
        class_mapping=class_names,
        nearest_nbors_search=True,
    )


# test sage model on PyG dataset directly
def test_sage_pyg() -> None:
    dataset_cora = Planetoid("./", "Cora")[0]
    model = SAGE(pyg_graph=dataset_cora, embedding_size=64)
    model._train(epochs=2)
    embeddings = model.get_embeddings()
    assert embeddings.size == int(173312)


# test gat model
def test_gat() -> None:
    metagraph = {
        "vertexCollections": {
            "cora_N": {"x": "x", "y": "y"},
        },
        "edgeCollections": {
            "cora_E": {},
        },
    }
    model = GAT(db, "graph", metagraph, embedding_size=64, heads=1)  # define model
    model._train(epochs=2)  # train
    embeddings = model.get_embeddings()  # get embeddings
    # check embeddings size
    assert embeddings.size == int(173312)


def test_m2v() -> None:
    metagraph = {
        "vertexCollections": {
            "movie": {"x": "x", "y": "y"},
            "director": {"x": "x"},
            "actor": {"x": "x"},
        },
        "edgeCollections": {
            "to": {},
        },
    }
    metapaths = [
        ("movie", "to", "actor"),
        ("actor", "to", "movie"),
    ]  # MAM # co-actor relationship

    model = METAPATH2VEC(
        db,
        "imdb",
        metagraph,
        metapaths,
        key_node="movie",
        embedding_size=4,
        walk_length=5,
        context_size=6,
        walks_per_node=5,
        num_negative_samples=5,
        sparse=True,
    )
    model._train(epochs=2, lr=0.03)
    embeddings = model.get_embeddings()
    # check embeddings size
    assert embeddings["movie"].size == int(17112)


def test_dmgi() -> None:
    metagraph = {
        "vertexCollections": {
            "movie": {"x": "x", "y": "y"},
            "director": {"x": "x"},
            "actor": {"x": "x"},
        },
        "edgeCollections": {
            "to": {},
        },
    }
    metapaths = [[("movie", "actor"), ("actor", "movie")]]  # MAM
    model = DMGI(db, "imdb", metagraph, metapaths, key_node="movie", embedding_size=4)
    model._train(epochs=2, lr=0.0005)
    embeddings = model.get_embeddings()
    # check embeddings size
    assert embeddings["movie"].size == int(17112)
    downstream_tasks.visualize_embeddings(
        model.G, embeddings["movie"], node_type="movie"
    )
