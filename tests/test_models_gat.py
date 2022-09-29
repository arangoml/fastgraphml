from .conftest import db
from graph_embeddings import GAT

def test_gat():
    metagraph = {
    "vertexCollections": {
            "Paper": {"x": "features", "y": "label"},
        },
        "edgeCollections": {
            "Cites": {},
        },
    }
    model = GAT(db, 'graph', metagraph, embedding_size=64)


    # assert embeddings.shape() == tensor([2708, 64]) ...


   