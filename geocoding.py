import pandas as pd
import json
from glob import glob
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import numpy as np

GEO_KEYS = ["continent", "subContinent", "country", "region", "city"]  # , "metro", "networkDomain"
NOT_AVAILABLE = "not available in demo dataset"
NOT_SET = "(not set)"


def extract_locations(f, geotree={}):
    data = pd.read_csv(f)

    for i in range(len(data)):
        geodata = json.loads(data.loc[i, ('geoNetwork')])

        for k in GEO_KEYS:
            if geodata[k] == NOT_AVAILABLE:
                geodata[k] = "None"
            elif geodata[k] == NOT_SET:
                geodata[k] = "NotSet"

        code = "_".join([geodata[k] for k in GEO_KEYS])

        # navigate through nested dictionaries
        current_geotree = geotree
        for i, k in enumerate(GEO_KEYS):
            if i == len(GEO_KEYS) - 1:
                current_geotree[geodata[k]] = code
            elif geodata[k] not in current_geotree:
                current_geotree[geodata[k]] = {}
            current_geotree = current_geotree[geodata[k]]

    return geotree

def geocode(s):
    geodata = json.loads(s)

    for k in GEO_KEYS:
        if geodata[k] == NOT_AVAILABLE:
            geodata[k] = "None"
        elif geodata[k] == NOT_SET:
            geodata[k] = "NotSet"

    return "_".join([geodata[k] for k in GEO_KEYS])

def build_graph(f, graph=None):
    if graph is None:
        graph = nx.Graph()

    data = pd.read_csv(f)

    for i in range(len(data)):
        code=geocode(data.loc[i, ('geoNetwork')])
        geodata[GEO_KEYS[-1]] = "LEAFNODE_" + code

        parent_key = "Planet Earth"
        weights = [2 ** i for i in reversed(range(len(GEO_KEYS)))]
        for i, (k, w) in enumerate(zip(GEO_KEYS, weights)):
            graph.add_edge(parent_key, geodata[k], weight=w)
            parent_key = geodata[k]

    return graph


if __name__ == "__main__":
    # geotree = {}
    #
    # for f in glob("train/*.csv"):
    #     extract_locations(f, geotree)
    #
    # for f in glob("test/*.csv"):
    #     extract_locations(f, geotree)
    #
    # json.dump(geotree, open("geotree.json", "w"))
    #
    # all_networks = {}
    #
    # for f in glob("train/*.csv"):
    #     data = pd.read_csv(f)
    #     for i in range(len(data)):
    #         geodata = json.loads(data.loc[i, ('geoNetwork')])
    #         domain = geodata["networkDomain"]
    #         if domain not in all_networks:
    #             all_networks[domain] = 1
    #         else:
    #             all_networks[domain] += 1
    #     break
    #
    # for key, value in sorted(all_networks.items(), key=lambda x: x[::-1]):
    #     print("%s: %s" % (key, value))

    graph = nx.Graph()
    for f in list(glob("train/*.csv")):
        graph = build_graph(f, graph)
    for f in list(glob("test/*.csv")):
        graph = build_graph(f, graph)

    write_dot(graph, "geodata.dot")

    distances = nx.floyd_warshall_numpy(graph)

    # TODO: extract columns corresponding to cities
    nodes = np.array(list(graph.nodes()), dtype=str)
    selection = np.array([n.startswith("LEAFNODE_") for n in graph.nodes()],dtype=bool)
    distances = distances[selection]
    distances = distances[:,selection]
    codes = nodes[selection]

    from sklearn import manifold
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=0,
                       dissimilarity="precomputed", n_jobs=1)
    X = mds.fit(distances).embedding_

    #fig = plt.figure(1)
    #ax = plt.axes([0., 0., 1., 1.])

    embedding = {codes[i][len("LEAFNODE_"):] : X[i].tolist() for i in range(X.shape[0])}
    json.dump(embedding, open("geocoding_embedding.json","w"))



    s = 100
    ax.scatter(X[:, 0], X[:, 1],X[:,2], color='navy', s=s, lw=0)
    plt.show()
