from setuptools import setup

with open("./README.md", encoding="utf-8") as fp:
    long_description = fp.read()


dev_requirements = [
    "black==22.8.0",
    "pytest==7.1.3",
    "sphinx==5.1.1",
    "pre-commit==2.20.0",
    "types-requests==2.28.11.2",
    "flake8==5.0.4",
    "pytest==7.1.3",
    "pytest-cov==4.0.0",
    "isort==5.10.1",
    "mypy==0.982",
    "types-setuptools==65.5.0.2",
    "bandit==1.7.4",
]


setup(
    name="fastgraphml",
    author="Sachin Sharma",
    author_email="sachin@arangodb.com",
    description="Given an input graph the library generates graph embeddings using Low-Code built on top of PyG",  # noqa: E501
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arangoml/fastgraphml",
    keywords=[
        "arangodb",
        "pyg",
        "graph deep learning",
        "graph embeddings",
        "pytorch geometric",
        "graph machine learning",
        "graph neural networks",
    ],
    packages=["fastgraphml"],
    include_package_data=True,
    python_requires=">=3.8",
    license="MIT License",
    install_requires=[
        "requests==2.28.1",
        "sklearn==0.0",
        "matplotlib==3.5.2",
        "seaborn==0.11.2",
        "matplotlib-inline==0.1.2",
        "numpy==1.22.4",
        "tqdm==4.64.0",
        "umap-learn==0.5.3",
        "adbpyg-adapter==1.1.0",
        "python-arango==7.4.1",
        "setuptools==45",
        "arango_datasets==1.0.0",
    ],
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Utilities",
        "Typing :: Typed",
    ],
)
