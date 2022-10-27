# configuration test
from adbpyg_adapter import ADBPyG_Adapter
from arango import ArangoClient
from arango.http import DefaultHTTPClient
from torch_geometric.datasets import Planetoid, IMDB


def pytest_addoption(parser) -> None:
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="")


def pytest_configure(config) -> None:
    global con
    con = {
        "url": config.getoption("url"),
        "username": config.getoption("username"),
        "password": config.getoption("password"),
        "dbName": config.getoption("dbName"),
    }

    print("----------------------------------------")
    print("URL: " + con["url"])
    print("Username: " + con["username"])
    print("Password: " + con["password"])
    print("Database: " + con["dbName"])
    print("----------------------------------------")

    class NoTimeoutHTTPClient(DefaultHTTPClient):  # type: ignore
        REQUEST_TIMEOUT = None

    global db
    db = ArangoClient(hosts=con["url"], http_client=NoTimeoutHTTPClient()).db(
        con["dbName"], con["username"], con["password"], verify=True
    )
    adbpyg = ADBPyG_Adapter(db)
    # loading cora dataset (Homogeneous)
    dataset_cora = Planetoid("./", "Cora")[0]
    if db.has_graph("cora"):
        db.delete_graph("cora", drop_collections=True, ignore_missing=True)
        adbpyg.pyg_to_arangodb("cora", dataset_cora)
    else:
        adbpyg.pyg_to_arangodb("cora", dataset_cora)

    # loading IMDB dataset (Heterogeneous)
    dataset_imdb = IMDB("./imdb")[0]
    if db.has_graph("imdb"):
        db.delete_graph("imdb", drop_collections=True, ignore_missing=True)
        adbpyg.pyg_to_arangodb("imdb", dataset_imdb)
    else:
        adbpyg.pyg_to_arangodb("imdb", dataset_imdb)
