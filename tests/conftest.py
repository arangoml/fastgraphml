# configuration test
from adbpyg_adapter import ADBPyG_Adapter
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.http import DefaultHTTPClient
from torch_geometric.datasets import Planetoid


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

    dataset = Planetoid("./", "Cora")
    data = dataset[0]
    # heterodata = data.to_heterogeneous(node_type_names=['Paper'], edge_types_names=[('Paper', 'Cites', 'Paper')])
    # db.delete_graph("Cora_Test", drop_collections=True, ignore_missing=True)
    adbpyg.pyg_to_arangodb(
        "Cora_Test", data, {"x": "features", "y": "label"}, overwrite=True
    )
