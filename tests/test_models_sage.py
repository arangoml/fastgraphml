from .conftest import db
from graph_embeddings import SAGE

# test sage model
def test_sage():
    metagraph = {
    "vertexCollections": {
            "cora_N": {"x": "x", "y": "y"},
        },
        "edgeCollections": {
            "cora_E": {},
        },
    }
    model = SAGE(db, 'graph', metagraph, embedding_size=64) # define model
    model._train(model, epochs=5) # train
    embeddings = model.get_embeddings(model=model) # get embeddings
    # check embeddings size
    assert embeddings.size == int(173312)


   