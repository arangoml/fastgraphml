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
        "matplotlib-inline",
        "seaborn",
        "adbpyg-adapter",
        "python-arango>=7.4.1",
        "setuptools>=45",
    ],
    extras_require={
        "dev": [
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