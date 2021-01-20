import networkx
import numpy as np
import scipy
import random
import pickle
import copy
import json

def graph_clean(graph, min_degree=-1):
    """
        clean the edge list by removing nodes that have no links
    :param graph:
    :param min_degree: removes a node with out_degree + in_degree < min_degree
    :return: cleaned_graph
    :rtype: dict
    """
    graph_undir = to_undirected(graph)
    is_valid = {}
    for node, neighbors in graph.items():
        if (node not in is_valid) and len(graph_undir[node]) >= min_degree:
            is_valid[node] = True
        for n in neighbors:
            if n not in is_valid and len(graph_undir[n]) >= min_degree:
                is_valid[n] = True

    graph_clean = {}
    for node, neighbors in graph.items():
        if node not in is_valid:
            continue
        if node not in graph_clean:
            graph_clean[node] = {}
        for n in neighbors:
            if n in is_valid:
                graph_clean[node][n] = copy.deepcopy(graph[node][n])
    return graph_clean

def graph_normalize(graph):
    """
            Normalizing id of the nodes
            e.g. 0 1 3 nodes would be 0 1 2 if node '2' is missing
        :param graph:
        :return: (normalized graph, new2old, old2new)
        :rtype: (dict, dict, dict)
        """
    old2new = {}
    new2old = {}
    new_id = 0
    for node, neighbors in graph.items():
        if node not in old2new:
            old2new[node] = str(new_id)
            new2old[old2new[node]] = node
            new_id += 1
        for n in neighbors:
            if n not in old2new:
                old2new[n] = str(new_id)
                new2old[old2new[n]] = n
                new_id += 1

    graph_normal = {}
    for node, neighbors in graph.items():
        node_new_id = old2new[node]
        if node_new_id not in graph_normal:
            graph_normal[node_new_id] = {}
        for n in neighbors:
            graph_normal[node_new_id][old2new[n]] = graph[node][n]
    return graph_normal, new2old, old2new


# load an edge list with weights
def graph_load(path, is_directed=True):
    graph_type = networkx.DiGraph() if is_directed else networkx.Graph()
    graph = networkx.readwrite.edgelist.read_weighted_edgelist(path, create_using=graph_type)
    graph = networkx.to_dict_of_dicts(graph)
    return graph

def node_save(node_num, path ):
    with open(path,'w') as f:
        for i in range(0,node_num+1):
            s = '{}\n'.format(i)
            f.write(s)

def generate_absent_links(graph):

    def find_graphsize(graph):
        gsize = 0
        for node, neighbors in graph.items():
            gsize = gsize + len(neighbors)
        return gsize

    def find_maxid(graph):
        maxid = 0
        node_list = []
        for node, neighbors in graph.items():
            tmp_neighbors = list(map(int,(neighbors.keys())))
            tmp_neighbors.append(int(node))
            maxid = max(maxid, max(tmp_neighbors))
            node_list = list(set(node_list) | set(tmp_neighbors))

        return maxid, node_list

    maxid, node_list = find_maxid(graph)
    graph_size = find_graphsize(graph)
    #rand_arr_initiators = np.random.randint(maxid, size = graph_size)
    rand_arr_initiators = np.random.choice(node_list, size = graph_size)
    unique, counts = np.unique(rand_arr_initiators, return_counts=True)
    rand_dic_initiators = dict(zip(unique, counts))
    
    rand_num = 0 
    rand_edges = {}
    for node, counts in rand_dic_initiators.items():
        #counts = (int)(graph_size / len(graph))
        connected_list = np.array([node])
        if str(node) in graph:
            connected_list = np.append(connected_list, list(map(int, graph[str(node)].keys())))

        rand_edges[node]={}
        for i in range(0, counts):
            tmp_rand_target = np.random.choice(node_list, size=1)[0]
            while tmp_rand_target in connected_list:
                tmp_rand_target = np.random.choice(node_list, size=1)[0]
            rand_edges[node][tmp_rand_target]=0
            connected_list = np.append(connected_list, tmp_rand_target)
            rand_num += 1

    print("%d gsize, %d rand num", graph_size, rand_num)

    return rand_edges

def graph_split_general(train_ratio, graph, task='link'):
    """
    Split the graph into train and test set
    :param train_ratio:
    :param graph:
    :return: (train_graph, test_graph)
    :rtype: (dict, dict)
    """
    # adds (node, neighbor) = value to graph
    def add(node, neighbor, value, mat):
        if node not in mat:
            mat[node] = {}
        mat[node][neighbor] = copy.deepcopy(value)

    def remove(node, neighbor, mat):
        if node in mat and neighbor in mat[node]:
            del mat[node][neighbor]

    if (task == 'link'):
        absent_edges = generate_absent_links(graph)

        for node, neighbors in absent_edges.items():
            for n in neighbors:
                add(str(node), str(n),  {'weight': absent_edges[node][n]}, graph)

    train_graph = {}
    train_graph_absent = {}
    test_graph = {}
    test_graph_absent = {}

    for node, neighbors in graph.items():
        for n in neighbors:
            if random.uniform(0, 1) < train_ratio:
                if graph[node][n]['weight'] == 0:
                    add(node, n, graph[node][n], train_graph_absent)
                else:
                    add(node, n, graph[node][n], train_graph)
                # delete the inverse if added to test data, re-add to train data
                if n in test_graph and node in test_graph[n]:
                    add(n, node, graph[n][node], train_graph)
                    remove(n, node, test_graph)
            else:
                if graph[node][n]['weight'] == 0:
                    add(node, n, graph[node][n], test_graph_absent)
                else:
                    add(node, n, graph[node][n], test_graph)
                # delete the inverse if added to train data, re-add to test data
                if n in train_graph and node in train_graph[n]:
                    add(n, node, graph[n][node], test_graph)
                    remove(n, node, train_graph)

    return train_graph, test_graph, train_graph_absent, test_graph_absent

# split given graph path to train and test
def graph_split(train_ratio, graph, new2old, old2new):

    """
    Split the graph into train and test set
    :param train_ratio:
    :param graph:
    :return: (train_graph, test_graph)
    :rtype: (dict, dict)
    """
    def add(node, neighbor, value, mat):
        if node not in mat:
            mat[node] = {}
        mat[node][neighbor] = copy.deepcopy(value)

    def remove(node, neighbor, mat):
        if node in mat and neighbor in mat[node]:
            del mat[node][neighbor]


    train_graph = {}
    test_graph = {}

    def translate_to_dummy( trgsouce, trgreceiver):
        dummy_id = old2new[new2old[trgsouce]["id"]]["out_dummy"]
        reverse_receiver = -1
        if new2old[trgreceiver]["type"] == "in+":
            reverse_receiver = old2new[new2old[trgreceiver]["id"]]["in-"]
        if new2old[trgreceiver]["type"] == "in-":
            reverse_receiver = old2new[new2old[trgreceiver]["id"]]["in+"]

        return dummy_id,reverse_receiver
    maxid = 0
    for node, neighbors in graph.items():
        maxid = max(maxid, int(node))
        if new2old[node]["type"] == "out_dummy":
            continue
        for n in neighbors:
            maxid = max(maxid, int(n))
            if random.uniform(0, 1) < train_ratio:
                add(node, n, graph[node][n], train_graph)
                dummy_source, dummy_receiver = translate_to_dummy(node, n)
                add(dummy_source, dummy_receiver, graph[dummy_source][dummy_receiver], train_graph)

                # delete the inverse if added to test data, re-add to train data
                #if n in test_graph and node in test_graph[n]:
                #    add(n, node, graph[n][node], train_graph)
                #    remove(n, node, test_graph)
            else:
                add(node, n, graph[node][n], test_graph)
                # delete the inverse if added to train data, re-add to test data
                #if n in train_graph and node in train_graph[n]:
                #    add(n, node, graph[n][node], test_graph)
                #    remove(n, node, train_graph)

    #add(maxid, maxid-1, {"weight" : 1}, train_graph)

    return train_graph, test_graph


def graph_select(graph, selector):
    """
        Return sub-graph of graph based on edges in selector
        Selector must be a sub-graph of graph
    :param graph:
    :param selector:
    :return:
    """
    selected = {}
    for node, neighbors in selector.items():
        if node not in selected:
            selected[node] = {}
        if node not in graph:
            raise Exception("node '%s' in selector but not in graph" % node)
        graph_neighbors = graph[node]
        for n in neighbors:
            if n not in graph_neighbors:
                raise Exception("link (%s, %s) in selector but not in graph" % (node, n))
            selected[node][n] = copy.deepcopy(graph[node][n])
    return selected


def graph_save(graph, path, is_directed=True):
    graph_type = networkx.DiGraph() if is_directed else networkx.Graph()
    networkx_graph = networkx.from_dict_of_dicts(graph, create_using=graph_type)
    networkx.readwrite.edgelist.write_weighted_edgelist(networkx_graph, path)


def json_save(dictionary, path):
    with open(path, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=2)


def json_load(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data




def graph_clean_type(graph, normalize=True):
    """
        Remove nodes that doesn't have all three link types
        (out), (in, +), (in, -)
    :param graph:
    :return: graph
    """
    # dictionary of marked (node, type) pairs to check missing link types
    marked = mark_3_types(graph)
    nodes2clean = {}
    for node, types in marked.items():
        for type, is_marked in types.items():
            if is_marked is False:
                nodes2clean[node] = True

    # remove nodes
    new_graph = {}
    for node, neighbors in graph.items():
        if node in nodes2clean:
            continue    # drop this node
        new_graph[node] = {}
        for n, attr in neighbors.items():
            if n in nodes2clean:
                continue    # drop this node
            new_graph[node][n] = copy.deepcopy(attr)

    # normalize ids to fill in the id gaps
    return graph_clean(new_graph) if normalize else new_graph


def mark_3_types(graph):
    """
        Mark visited (old_id, type) pairs like (1, out), (1, in+)
    :param graph:
    :return: marked (node, type) pairs
    :rtype: dict
    """
    marked = {}  # marked[old id][out / in+ / in-] = True / False

    def mark(old_id, type, marked):
        if old_id not in marked:
            marked[old_id] = {"out": False, "in+": False, "in-": False}
        marked[old_id][type] = True

    for node, neighbors in graph.items():
        if len(neighbors) == 0:
            continue
        mark(old_id=node, type="out", marked=marked)
        for n, attr in neighbors.items():
            if attr['weight'] > 0:
                mark(old_id=n, type="in+", marked=marked)
            elif attr['weight'] < 0:
                mark(old_id=n, type="in-", marked=marked)
    return marked


def to_3types(graph):
    """
        Transform directed, signed graph to undirected unsigned graph
        by expanding each node to three link types
        (out), (in, +), (in, -)
    :param graph
    :return: (graph, new2old, old2new)
    :rtype: (dict, dict, dict)
    """
    new_graph = {}      # new_graph[new id 1][new id 2] = {weight: -1}
    new2old = {}        # new2old[new id] = {id: old id, type: out / in+ / in-}
    old2new = {}        # old2new[old id][out / in+ / in-] = new id
    new_id = 0          # current new id counter

    # Mark visited (old_id, type) pairs like (1, out), (1, in+)
    # New ids are assigned after a marking phase
    # to assign sequential ids i/i+1/i+2 to out/in+/in- links of a node
    marked = mark_3_types(graph)

    # assign new id to visited (node, type) pairs

    def map(old_id, type, new_id, old_new, new_old):
        new_id_str = str(new_id)
        if old_id not in old_new:
            old_new[old_id] = {type: new_id_str}
        elif type in old_new[old_id]:
            raise Exception("duplicate (node, type) pair")
        else:
            old_new[old_id][type] = new_id_str
        new_old[new_id_str] = {"id": old_id, "type": type}
        return new_id + 1

    for node, types in marked.items():
        for type, is_marked in types.items():
            if is_marked:
                new_id = map(old_id=node, type=type, new_id=new_id, old_new=old2new, new_old=new2old)

    # build the new graph

    # adds (v1, v2) = value to graph
    def add(v1, v2, weight, mat):
        if v1 not in mat:
            mat[v1] = {}
        mat[v1][v2] = {'weight': weight}

    for node, neighbors in graph.items():
        for n, attr in neighbors.items():
            weight = attr['weight']
            if weight > 0:
                add(old2new[node]["out"], old2new[n]["in+"], weight, new_graph)
            elif weight < 0:
                add(old2new[node]["out"], old2new[n]["in-"], -weight, new_graph)

    return new_graph, new2old, old2new


def to_3types_full(graph, old2new=None, augment_net = True):

    """
        This transformation differs from to_3types in that even non-existent node-types
        receive an id,
        e.g. if node 2 receives no positive link, 2in- still occupies a new id
        Also old2new mapper can be fed from an already conversion instead of building a new one
    :param graph
    :param old2new: convert the graph with the given map id (do not build a new one
    :return: (graph, new2old, old2new)
    :rtype: (dict, dict, dict)
    """

    new_graph = {}     # new_graph[new id 1][new id 2] = {weight: -1}
    new2old = {}        # new2old[new id] = {id: old id, type: out / in+ / in-}
    old2new = {} if old2new is None else old2new   # old2new[old id][out / in+ / in-] = new id
    new_id = 0          # current new id counter

    if len(old2new) == 0:
        # assign new id to all types of visited nodes
        def map(old_id, new_id, old_new, new_old):
            if old_id not in old_new:

                new_id_out = str(new_id)
                new_id_inp = str(new_id + 1)
                new_id_inn = str(new_id + 2)
                new_id_out_dummy = str(new_id + 3)
                old_new[old_id] = {'out': new_id_out, 'in+': new_id_inp, 'in-': new_id_inn, 'out_dummy': new_id_out_dummy}

                new_old[new_id_out] = {"id": old_id, "type": "out"}
                new_old[new_id_inp] = {"id": old_id, "type": "in+"}
                new_old[new_id_inn] = {"id": old_id, "type": "in-"}
                new_old[new_id_out_dummy] = {"id": old_id, "type": "out_dummy"}

                return new_id + 4
            else:
                return new_id

        for node, neighbors in graph.items():
            new_id = map(old_id=node, new_id=new_id, old_new=old2new, new_old=new2old)
            for n, attr in neighbors.items():
                new_id = map(old_id=n, new_id=new_id, old_new=old2new, new_old=new2old)

    # build the new graph

    # adds (v1, v2) = value to graph
    def add(v1, v2, weight, mat):
        if v1 not in mat:
            mat[v1] = {}
        mat[v1][v2] = {'weight': weight}

    for node, neighbors in graph.items():
        for n, attr in neighbors.items():
            weight = attr['weight']
            if weight > 0:
                add(old2new[node]["out"], old2new[n]["in+"], weight, new_graph)
                add(old2new[node]["out_dummy"], old2new[n]["in-"], weight, new_graph)
            elif weight < 0:
                add(old2new[node]["out"], old2new[n]["in-"], -weight, new_graph)
                add(old2new[node]["out_dummy"], old2new[n]["in+"], -weight, new_graph)

    return new_graph, new2old, old2new

def to_undirected(graph):
    """
        If A -> B, B -> A is also created
    :param graph:
    :return:
    """
    new_graph = {}
    for node, neighbors in graph.items():
        if node not in new_graph:
            new_graph[node] = {}
        for n, attr in neighbors.items():
            if n not in new_graph:
                new_graph[n] = {}
            new_graph[node][n] = copy.deepcopy(attr)
            new_graph[n][node] = copy.deepcopy(attr)
    return new_graph


def three_type_to_graph(graph_3type, new2old):
    graph = {}
    for node, neighbors in graph_3type.items():
        source = str(new2old[node]["id"])
        if source not in graph:
            graph[source] = {}
        for n, attr in neighbors.items():
            target = str(new2old[n]["id"])
            weight = int(attr["weight"]) if new2old[n]["type"] == "in+" else -int(attr["weight"])
            graph[source][target] = {"weight": weight}
    return graph


def apply_on_weight(graph: dict, func):
    """
        Apply the function on each weight
    :return: graph
    :rtype: dict
    """
    new_graph = {}
    for node, neighbors in graph.items():
        if node not in new_graph:
            new_graph[node] = {}
        for n, attr in neighbors.items():
            new_attr = copy.deepcopy(attr)
            new_attr["weight"] = func(attr["weight"])
            new_graph[node][n] = new_attr
    return new_graph

def get_link_meta_3_types(source, target, new2old, old2new):
    """
        Return meta data of node pairs from original graph before transformation
    :param source: id of source node in transformed graph
    :param target:
    :param new2old: map of new ids to old (id, type) before transform
    :return: source, target_in+, target_in-, weight
    :rtype: (int, int, int, int)
    """

    if (source not in new2old) or ((new2old[source]["type"] != "out") and (new2old[source]["type"] != "out_dummy")):

        raise Exception("node '%s' is not registered as 'out' but as '%s'" % (source,new2old[source]["type"]))

    if new2old[source]["type"] == "out_dummy":
        return -1, -1, -1, -1, -1

    target_old = str(new2old[target]["id"])
    weight = -1 if new2old[target]["type"] == "in-" else 1
    target_out = old2new[target_old]["out"]
    if weight > 0:
        target_in_pos = old2new[target_old]["in+"] if "in+" in old2new[target_old] else -1
        target_in_neg = old2new[target_old]["in-"] if "in-" in old2new[target_old] else -1
    else:
        target_in_pos = old2new[target_old]["in+"] if "in+" in old2new[target_old] else -1
        target_in_neg = old2new[target_old]["in-"] if "in-" in old2new[target_old] else -1

    return int(source), int(target_in_pos), int(target_in_neg), int(target_out), weight


def to_2types(graph):
    """
        Transform directed, signed graph to undirected unsigned graph
        of 2nd order relations.
        For example, if (A -> B+, A -> C-) and (D -> B+, D -> C-), then w(B+, C-) = 2
    :return: (graph, new2old, old2new)
    :rtype: (dict, dict, dict)
    """
    graph_3types, new2old, old2new = to_3types_full(graph)
    g = networkx.from_dict_of_dicts(graph_3types, create_using=networkx.DiGraph())   # directed
    m = networkx.convert_matrix.to_scipy_sparse_matrix(G=g, nodelist=new2old.keys())
    m.data = np.ones(shape=m.data.shape)    # change graph to unweighted
    mt = scipy.transpose(m)
    m2 = mt.dot(m)
    m2.setdiag(0)
    m2.eliminate_zeros()
    # convert m2 type from csc_matrix to lil_matrix for efficiency
    graph_2types = node_int_to_string(networkx.to_dict_of_dicts(
        networkx.convert_matrix.from_scipy_sparse_matrix(m2.tolil())))
    return graph_2types, new2old, old2new


def normalize_3types_to_2types(graph, new2old, old2new):
    """
        Removes Nout indices and re-normalizes Nin- and Nin+ indices
    :param graph: input graph that has no Nout index
    :param new2old: a map that connect 3types ids to original ids
    :param old2new: a map that connects original ids to 3types ids
    :return:
    """


def graph_statistics(graph):
    """
    :param graph:
    :return: node_count, edge_count, negative_edge_count, maximum node id
    :rtype: (int, int, int, int)
    """
    visited_nodes = {}
    node_count = 0
    edge_count = 0
    negative_edge_count = 0
    max_id = -1
    for node, neighbors in graph.items():
        max_id = int(node) if int(node) > max_id else max_id
        if node not in visited_nodes:
            visited_nodes[node] = True
            node_count += 1
        for n, attr in neighbors.items():
            max_id = int(n) if int(n) > max_id else max_id
            if n not in visited_nodes:
                visited_nodes[n] = True
                node_count += 1
            weight = attr['weight']
            if weight != 0:
                edge_count += 1
                if weight < 0:
                    negative_edge_count += 1
    return node_count, edge_count, negative_edge_count, max_id


def node_int_to_string(graph):
    """
        Convert int node ids to string
    :param graph:
    :return:
    """
    new_graph = {}
    for node, neighbors in graph.items():
        node_str = str(node)
        if node_str not in new_graph:
            new_graph[node_str] = {}
        for n, attr in neighbors.items():
            new_graph[node_str][str(n)] = copy.deepcopy(attr)
    return new_graph


def to_weighted(graph, weight=1):
    """
        Add a default weight to edges of graph if it doesn't have any
    :param graph:
    :param weight:
    :return:
    """
    new_graph = {}
    for node, neighbors in graph.items():
        if node not in new_graph:
            new_graph[node] = {}
        for n, attr in neighbors.items():
            new_graph[node][n] = {'weight': weight}
    return new_graph

