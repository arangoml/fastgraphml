from .conftest import db
from graph_embeddings import SAGE

def test_gat():
    metagraph = {
    "vertexCollections": {
            "Paper": {"x": "features", "y": "label"},
        },
        "edgeCollections": {
            "Cites": {},
        },
    }
    model = SAGE(db, 'graph', metagraph, embedding_size=64)
    model._train(model, epochs=5) # train
    embeddings = model.get_embeddings(model=model) # get embeddings

    assert embeddings.size == int(173312)


   