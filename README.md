# fastgraphml
Given an input graph it generates Graph Embeddings using Low-Code framework built on top of [PyG](https://pytorch-geometric.readthedocs.io/en/latest/). The package supports training on both GPU and CPU enabled machines. Training jobs on GPUs results in much faster execution and increased in performance when it comes to handling large graphs as compared to CPUs. In addition, the framework provides tight integration with  [ArangoDB](https://www.arangodb.com/) which is a scalable, fully managed graph database, document store and search engine in one place. Once Graph Embeddings are generated, they can be used for various downstream machine learning tasks like Node Classification, Link Prediction, Visualisation, Community Detection, Similartiy Search, Recommendation, etc. 

## Installation
#### Required Dependencies
1. PyTorch `1.12.*` is required.
    * Install using previous version that matches your CUDA version: [pytorch](https://pytorch.org/get-started/previous-versions/)
        * To find your installed CUDA version run `nvidia-smi` in your terminal.
2. [pyg](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
3. [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
    * Note: For FAISS-CPU one needs `numba==0.53.0`

#### Latest Release
```
pip install fastgraphml
```

## Quickstart: Graph Embedding Generation

### Use Case 1: Generates Graph Embeddings using the graphs stored inside ArangoDB:

#### Example Homogneous Graphs

```python
from fastgraphml.graph_embeddings import SAGE, GAT
from fastgraphml.graph_embeddings import downstream_tasks
from fastgraphml import Datasets 
from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient("http://127.0.0.1:8529")
db = client.db('_system', username='root', password='')

# Loading Amazon Computer Products dataset into ArangoDB
Datasets(db).load("AMAZON_COMPUTER_PRODUCTS")

# Optionally use arangodb graph
# arango_graph = db.graph('product_graph')

# metadata information of arango_graph
metagraph = {
    "vertexCollections": {
        "Computer_Products": {"x": "features", "y": "label"},
    },
    "edgeCollections": {
        "bought_together": {},
    },
}

# generating graph embeddings with 3 lines of code
model = SAGE(db,'product_graph', metagraph, embedding_size=64) # define graph embedding model
model._train(epochs=10) # train
embeddings = model.get_embeddings() # get embeddings
```

#### Example Heterogeneous Graphs

```python
from fastgraphml.graph_embeddings import METAPATH2VEC, DMGI
from fastgraphml.graph_embeddings import downstream_tasks 
from fastgraphml import Datasets 

from arango import ArangoClient

# Initialize the ArangoDB client.
client = ArangoClient("http://127.0.0.1:8529")
db = client.db('_system', username='root')

# Loading IMDB Dataset into ArangoDB
Datasets(db).load("IMDB_X")

# Optionally use ArangoDB Graph
# arango_graph = db.graph("IMDB")

metagraph = {
    "vertexCollections": {
    
        "movie": { "x": "x", "y": "y"},  
        "director": {"x": "x"},
        "actor": {"x": "x"},
    },
    "edgeCollections": {
        "to": {},
    },
}
metapaths = [('movie', 'to','actor'),
             ('actor', 'to', 'movie'), ] # MAM # co-actor relationship

# generating graph embeddings with 3 lines of code
model = METAPATH2VEC(db, "IMDB_X", metagraph, metapaths, key_node='movie', embedding_size=128,
                     walk_length=5, context_size=6, walks_per_node=5, num_negative_samples=5,
                     sparse=True) # define model
model._train(epochs=10, lr=0.03) # train
embeddings = model.get_embeddings() # get embeddings
```

### Use Case 2: Generates Graph Embeddings using PyG graphs:

```python
from fastgraphml.graph_embeddings import SAGE, GAT
from fastgraphml.graph_embeddings import downstream_tasks 
from torch_geometric.datasets import Planetoid

# load pyg dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# generating graph embeddings with 3 lines of code
model = SAGE(pyg_graph=data, embedding_size=64) # define graph embedding model
model._train(epochs=10) # train
embeddings = model.get_embeddings() # get embeddings
```
## Models Supported

Model         | Homogeneous   | Heterogeneous | Node Features
------------- | ------------- | ------------- | ------------- 
[GraphSage](https://arxiv.org/abs/1706.02216)     | ✔️             |               | ✔️ 
[GAT](https://arxiv.org/abs/1710.10903)           | ✔️             |               | ✔️ 
[Metapath2Vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)  |               | ✔️             |  
[DMGI](https://arxiv.org/pdf/1911.06750.pdf)          |               | ✔️             | ✔️ 



## Quickstart: Downstream Tasks
In addition, the library also provides various low-code helper methods to carry out number of downstream tasks such as visualisation, similarity search (recommendation) , and link prediction (to be added soon).

### Downstream Task 1: Graph Embedding Visualisation
This method helps in visualization of generated Graph Embeddings by reducing them 2 dimensions using U-Map.
#### Example
```python
# amazon computers dataset
class_names = {0: 'Desktops',1: 'Data Storage',2: 'Laptops',3: 'Monitors',4: 'Computer Components',
 5: 'Video Projectors',6: 'Routers',7: 'Tablets',8: 'Networking Products',9: 'Webcams'}
# with one line of code
downstream_tasks.visualize_embeddings(model.G, embeddings, class_mapping=class_names, emb_percent=0.1) # model.G is PyG data object
```
### Downstream Task 2: Scalable Similarity Search with Faiss
[Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) is a tool developed by Facebook that performs similarity search in sets of vectors of any size, up to ones that possibly do not fit in RAM.
We support two types of search for now:
1. exact search: For precise similarity search but at the cost of scalability.
2. approx search: For scalable similarity search but at the cost of some precision loss.
#### Example 1
```python
downstream_tasks.similarity_search(embeddings, top_k_nbors=10, nlist=10, search_type='exact')
```
#### Example 2
If nearest_nbors_search=True, store_embeddings method saves generated Graph Embeddings in ArangoDB along with top_k nearest neighbors (node ids with similar embeddings) and their corresponding similarity scores (i.e. cosine distance). 
```python
model.graph_util.store_embeddings(embeddings, collection_name=None, batch_size=100, class_mapping=None, 
        nearest_nbors_search=False, top_k_nbors=10, nlist=10, search_type='exact')
```

