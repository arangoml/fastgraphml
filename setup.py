from setuptools import setup

with open("./README.md") as fp:
    long_description = fp.read()

setup(
    name="fastgraphml",
    author="Sachin Sharma",
    author_email="sachin@arangodb.com",
    description="Given an input graph the libraray generates graph embeddings using Low-Code built on top of PyG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arangoml/fastgraphml",
    keywords=["arangodb", "pyg", "graph deep learning" ,"graph embeddings", "pytorch geometric",
    "graph machine learning", "graph neural networks"],
    packages=["graph_embeddings"],
    include_package_data=True,
    python_requires=">=3.8",
    license="MIT License",
    install_requires=[
        "numpy",
        "sklearn",
        "tqdm",
        "umap-learn",
        "matplotlib",
        "numba==0.53.0", # for CPU (used by FAISS)
        "matplotlib-inline",
        "seaborn",
        "torch>=1.12.0",
        "torch-cluster==1.6.0",
        "torch-sparse>=0.6.14",
        "torch-scatter>=2.0.9",
        "torch-geometric>=2.0.4",
        "adbpyg-adapter",
        "python-arango>=7.4.1",
        "setuptools>=45",
    ],
    extras_require={
        "dev": [
            "black==22.8.0",
            "mypy==0.981",
            "flake8==5.0.4",
            "isort==5.10.1"
        ],
    },
    classifiers=[
        "Intended Audience :: Data Scientists/ML Engineers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
)
