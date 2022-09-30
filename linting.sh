#/bin/sh
# formatting first
isort .
black ./
# then linting
# mypy ./
# flake8
