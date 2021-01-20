import settings
import const
import os
import re
import subprocess
from src.methods.node2vec import Node2Vec
import networkx

def graph_load(path, is_directed=False):
    graph_type = networkx.DiGraph() if is_directed else networkx.Graph()
    graph = networkx.readwrite.edgelist.read_weighted_edgelist(path, create_using=graph_type)
    return graph

def embed_py(graph_path, output_path, is_windows=True, is_directed=False, dimension=20, walk_length=60, nb_of_walk_per_srouce = 12, window_size = 5):
    graph = graph_load(graph_path, is_directed)
    node2vec = Node2Vec(graph, dimension, walk_length, nb_of_walk_per_srouce, workers=1)
    model = node2vec.fit(window=window_size, min_count=1, batch_words=4)
    model.wv.save_word2vec_format(output_path)

def embed_cpp(graph_path, output_path, is_windows=True, is_directed=False, dimension=20, walk_length=60, nb_of_walk_per_srouce = 12, p=1, q=1):
    # node2vec -i:graph/karate.edgelist -o:emb/karate.emb -l:3 -d:24 -p:0.3 -dr -v
    node2vec_path = settings.config[const.NODE_TO_VEC]
    command_list = [win_to_cygwin(node2vec_path) if is_windows else node2vec_path,
                    '-i:' + win_to_cygwin(graph_path) if is_windows else graph_path,
                    '-o:' + win_to_cygwin(output_path) if is_windows else output_path,
                    '-d:%d' % dimension,
                    '-p:%f' % p, '-q:%f' % q, '-l:%d' % walk_length, '-r:%d' % nb_of_walk_per_srouce ,  '-w', '-v']
    if is_directed:
        command_list.append("-dr")
    command = " ".join(command_list)
    if is_windows:
        # Call the configured shell from windows cmd to run node2vec.exe
        command = "%s -l -c \"%s\"" % (settings.config[const.SHELL], command)
    print('Execute: %s' % command)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
    return p.wait()

def win_to_cygwin(path: str):
    return "/cygdrive/" + path[0:1].lower() + path[2:].replace("\\", "/")

if __name__ == '__main__':
    embed_cpp(graph_path=settings.config[const.SLASHDOT_3TYPE_TRAIN],
              output_path=settings.config[const.SLASHDOT_3TYPE_OUTPUT],
              is_directed=False)
    embed_cpp(graph_path=settings.config[const.SLASHDOT_UNSIGNED_TRAIN],
              output_path=settings.config[const.SLASHDOT_UNSIGNED_OUTPUT],
              is_directed=False)
