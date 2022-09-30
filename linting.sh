#/bin/sh
# formatting first
isort .
black ./
# then linting
mypy ./graph_embeddings
flake8 ./graph_embeddings
