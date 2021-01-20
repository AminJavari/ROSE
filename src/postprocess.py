
from scipy import io
import pickle
import numpy as np
import random
from src import preprocess
from src import util
import copy
import pandas as pd
import os


def n2v_sgcn_save_embeddings_to_mat(model_path, save_path):
    data = pd.read_csv(model_path+'emb.csv', delimiter=",", header=None, skiprows=1)
    data.sort_values(by=data.columns[0], inplace=True)
    matrix = util.df2array(df=data)
    io.savemat(save_path, {'emb': matrix})

def n2v_save_embeddings_to_mat(embed_path, save_path):
    data = pd.read_csv(embed_path, delimiter=" ", header=None, skiprows=1)
    data.sort_values(by=data.columns[0], inplace=True)
    matrix = util.df2array(df=data)
    io.savemat(save_path, {'emb': matrix})

def get_node_embedding(node, embed):
    node_embed = np.zeros(embed.shape[1])
    if int(node) < len(embed):
        node_embed = embed[int(node) , :]
    return node_embed


def generate_dataset_3type(graph, graph_absent, embed_path,
                           new2old_path, old2new_path, exclude_missing=False,
                           relations_path="",
                           unsigned_embed_path="",
                           save_path="", similarity_type="distance",
                           sample_rate=1, neg2pos_ratio=1, task='link'
                           ):
    """
        Each row: Embed(A) | Embed(Aout) | Embed(Bin-) | Embed(Bin+) | sign(A, B)
    :param graph:
    :param embed_path:
    :param unsigned_embed_path:
    :param relations_path: graph of neighbors (preferably test + train)
    :param new2old_path:
    :param old2new_path:
    :param save_path:
    :param similarity_type: distance or product
    :param sample_rate:
    :param neg2pos_ratio: if < 1 positive links will be under-sampled to balance the dataset
    :param exclude_missing: reject links with missing embeddings
    :return: dataset
    :rtype: dict
    """
    print('Similarity type:', similarity_type)
    embed = io.loadmat(embed_path)['emb']

    if len(unsigned_embed_path) > 0 and len(relations_path) > 0:
        embed_unsigned = io.loadmat(unsigned_embed_path)['emb']  # embedding of unsigned graph
        relations = preprocess.graph_load(relations_path)  # neighbors (preferably test + train)
    else:
        embed_unsigned = None
        relations = None

    new2old = preprocess.json_load(new2old_path)    # map of node ids in transformed graph to old (node, type) pairs
    old2new = preprocess.json_load(old2new_path)    # map of old ids to new ids per type
    dataset = []
    # metadata[:, 0] = source node id from original graph
    # metadata[:, 1] = target node id from original graph
    # metadata[:, 2] = 1 if in+ exists, 0 o.w.,
    # metadata[:, 3] = 1 if in- exists, 0 o.w.
    metadata = []
    positive_samples = 0
    negative_samples = 0
    original_samples = 0
    for node, neighbors in graph.items():
        for target, attr in neighbors.items():

            source, target_pos, target_neg, target_out, weight = preprocess.get_link_meta_3_types(
                source=node, target=target, new2old=new2old, old2new=old2new)

            #dont use dummy nodes in training/test
            if source == -1:
                continue
            original_samples += 1
            _, source_pos, source_neg, _, _ = preprocess.get_link_meta_3_types(
                source=str(source), target=str(source), new2old=new2old, old2new=old2new)

            rand = random.uniform(0, 1)
            # P(select positive) = P(select negative) * negative ratio
            accept = (weight > 0 and rand < (sample_rate * neg2pos_ratio)) \
                     or (weight < 0 and rand < sample_rate)

            if accept:
                positive_samples = positive_samples + 1 if weight > 0 else positive_samples
                negative_samples = negative_samples + 1 if weight < 0 else negative_samples

                source_out_embed = embed[source, :]
                target_out_embed = embed[target_out, :]
                # ID of Ain+ or Ain- may be outside of embed size
                # since the node2vec input didn't have any link toward them
                source_pos_embed = embed[source_pos, :] \
                    if source_pos < embed.shape[0] else np.zeros(embed.shape[1])
                source_neg_embed = embed[source_neg, :] \
                    if source_neg < embed.shape[0] else np.zeros(embed.shape[1])

                target_pos_embed = embed[target_pos, :]
                target_neg_embed = embed[target_neg, :]
                has_target_pos = 1 * np.any(target_pos_embed)
                has_target_neg = 1 * np.any(target_neg_embed)
                has_target_out = 1 * np.any(target_out_embed)

                if has_target_out == 0:
                    target_out_embed = np.zeros(embed.shape[1])

                if has_target_pos == 0:
                    target_pos_embed = np.ones(embed.shape[1])*0

                if has_target_neg == 0:
                    target_neg_embed = np.ones(embed.shape[1])*0

                link_sign = np.asarray([float(weight)], dtype='float32')
                source_old = int(new2old[node]['id'])
                target_old = int(new2old[target]['id'])
                # In training, if positive or negative type
                # is not embedded for target node, reject the link
                if exclude_missing and \
                        (has_target_pos == 0 or has_target_neg == 0):
                    continue
                # the order of elements in row_meta is important for sign prediction
                row_meta = np.array([source_old, target_old,
                                     has_target_pos, has_target_neg])

                metadata.append(row_meta)
                if relations is not None:
                    # find source_embed as a weighted average of neighbor embeddings
                    # source_embed = sum_n W(neighbor_n_embed, target_embed) * neighbor_n_(pos/neg)_embed
                    neighbors_no_target = copy.deepcopy(relations[node])
                    neighbors_no_target.pop(target, None)
                    source_embed = self_attention(target, neighbors_no_target,
                                                  embed, embed_unsigned, new2old,
                                                  type=similarity_type)
                    # A|B, Bin+, Bin-
                    row = np.concatenate((source_embed, source_pos_embed, source_neg_embed, target_out_embed,
                                          target_pos_embed, target_neg_embed,  link_sign))
                else:
                    row = np.concatenate((source_out_embed, source_pos_embed, source_neg_embed, target_out_embed,
                                          target_pos_embed, target_neg_embed, link_sign))
                dataset.append(row)
    total_samples = positive_samples + negative_samples
    absent_samples = 0
    total_absent = 0
    if task == 'link':
        selection_prob = 1 - (total_samples/original_samples)
        for node, neighbors in graph_absent.items():
            if str(node) in old2new:
                source = int(old2new[str(node)]["out"])
                source_pos = int(old2new[str(node)]["in+"])
                source_neg = int(old2new[str(node)]["in-"])
                source_pos_embed = embed[source_pos, :] \
                    if source_pos < embed.shape[0] else np.zeros(embed.shape[1])
                source_neg_embed = embed[source_neg, :] \
                    if source_neg < embed.shape[0] else np.zeros(embed.shape[1])
                source_out_embed = embed[source, :] \
                    if source < embed.shape[0] else np.zeros(embed.shape[1])
            else:
                source_pos_embed = np.zeros(embed.shape[1])
                source_neg_embed = np.zeros(embed.shape[1])
                source_out_embed = np.zeros(embed.shape[1])

            for n in neighbors:
                total_absent +=1
                if random.uniform(0, 1) > selection_prob:
                    absent_samples += 1
                    if str(n) in old2new:
                        target_pos = int(old2new[str(n)]["in+"])
                        target_neg = int(old2new[str(n)]["in-"])
                        target_out = int(old2new[str(n)]["out"])
                        target_pos_embed = embed[target_pos, :] \
                            if target_pos < embed.shape[0] else np.zeros(embed.shape[1])
                        target_neg_embed = embed[target_neg, :] \
                            if target_neg < embed.shape[0] else np.zeros(embed.shape[1])
                        target_out_embed = embed[target_out, :] \
                            if target_out < embed.shape[0] else np.zeros(embed.shape[1])
                    else:
                        target_pos_embed = np.zeros(embed.shape[1])
                        target_neg_embed = np.zeros(embed.shape[1])
                        target_out_embed = np.zeros(embed.shape[1])

                    link_sign = np.asarray([float(0)], dtype='float32')

                    row_meta = np.array([int(node), int(n),1,1])
                    if relations is not None:
                    # find source_embed as a weighted average of neighbor embeddings
                    # source_embed = sum_n W(neighbor_n_embed, target_embed) * neighbor_n_(pos/neg)_embed
                        neighbors_no_target = copy.deepcopy(relations[node])
                        neighbors_no_target.pop(target, None)
                        source_embed = self_attention(target, neighbors_no_target,
                                                  embed, embed_unsigned, new2old,
                                                  type=similarity_type)
                    # A|B, Bin+, Bin-
                        row = np.concatenate((source_embed, source_pos_embed, source_neg_embed, target_out_embed,
                                          target_pos_embed, target_neg_embed,  link_sign))
                    else:
                        row = np.concatenate((source_out_embed, source_pos_embed, source_neg_embed, target_out_embed,
                                          target_pos_embed, target_neg_embed, link_sign))
                    #row = np.concatenate((source_out_embed, source_pos_embed, source_neg_embed, target_out_embed,
                    #                      target_pos_embed, target_neg_embed, link_sign))
                        dataset.append(row)
                        metadata.append(row_meta)

    dataset = np.vstack(dataset)    # convert the list of rows to a 2d array
    metadata = np.vstack(metadata)
    dataset_all = {'dataset': dataset, 'metadata': metadata}
    if len(save_path) > 0:
        io.savemat(save_path, dataset_all)
    # print statistics
    excluded_samples_rate = (total_samples - dataset.shape[0]) / total_samples
    missing_samples = sum((metadata[:, 2] == 0) | (metadata[:, 3] == 0))
    missing_samples_rate = missing_samples / dataset.shape[0]
    print("%d links sampled (%d positive, %d negative), %d absent link, %d total abset, "
          "%.2f%% excluded, now %.2f%% has missing embeddings"
          % (total_samples, positive_samples, negative_samples, absent_samples, total_absent,
             excluded_samples_rate * 100, missing_samples_rate * 100))
    return dataset_all


def self_attention(target, source_neighbors, embed, embed_unsigned, new2old, type="distance"):
    """
        Weight each neighbor of source
        according to distance of neighbor to target node in unsigned graph
        Then return the weighted sum of neighbor's embeddings
    :param target:
    :param source_neighbors:
    :param embed:
    :param embed_unsigned:
    :param new2old:
    :param type: distance or product
    :return:
    """
    if target in source_neighbors:
        raise Exception("Known (source, target) relation cannot be used, it's cheating!")
    weight = {}
    embed_result = np.zeros(shape=(embed.shape[1]))
    target_old = int(new2old[target]["id"])
    total_weight = 0
    for n in source_neighbors:
        n_old = int(new2old[n]["id"])
        if type == "distance":
            distance = np.linalg.norm(embed_unsigned[target_old, :] - embed_unsigned[n_old, :])
            weight[n] = np.exp(-distance)
        elif type == "product":
            product = np.inner(embed_unsigned[target_old, :], embed_unsigned[n_old, :])
            weight[n] = np.exp(product)
        total_weight += weight[n]
    # normalize weight of neighbors to sum() = 1
    for n in source_neighbors:
        embed_result += (weight[n] / total_weight) * embed[int(n), :]
    return embed_result

def generate_absent_links(graph_main, graph_side):
    def find_graphsize(graph):
        gsize = 0
        for node, neighbors in graph_main.items():
            gsize = gsize + len(neighbors)
        return gsize

    def find_maxid(graph):
        maxid = 0
        for node, neighbors in graph_main.items():
            tmp_neighbors = list(map(int,(neighbors.keys())))
            tmp_neighbors.append(int(node))
            maxid = max(maxid, max(tmp_neighbors))

        return maxid

    maxid = max(find_maxid(graph_main), find_maxid(graph_side))
    graph_size = find_graphsize(graph_main)
    rand_arr_initiators = np.random.randint(maxid, size = graph_size)
    unique, counts = np.unique(rand_arr_initiators, return_counts=True)
    rand_dic_initiators = dict(zip(unique, counts))

    rand_edges = {}
    for node, counts in rand_dic_initiators.items():
        connected_list = np.array([node])
        if str(node) in graph_main:
            connected_list = np.append(connected_list, list(map(int, graph_main[str(node)].keys())))
        if str(node) in graph_side:
            connected_list = np.append(connected_list, list(map(int, graph_side[str(node)].keys())))

        rand_edges[node]={}
        for i in range(0, counts):
            tmp_rand_target = random.randint(1, maxid)
            while tmp_rand_target in connected_list:
                tmp_rand_target = random.randint(1, maxid)
            rand_edges[node][tmp_rand_target]=0

    return rand_edges


