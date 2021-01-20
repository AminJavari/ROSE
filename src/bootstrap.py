import sys
import os
import argparse
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter
from src import preprocess as pre
from src import postprocess as post
from src import sign_prediction
from src import util
from pathlib import Path
from src.methods import sine, mynode2vec
from src.methods.side import side
from src.methods.beside import BESIDE_train
#from src.methods.sgcn.src import mysgcn
import const

def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    print(sys.argv[1:])
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def get_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='commands')
    # ------ Embed and build sign prediction model ------ #
    embed_parser = commands.add_parser('build',
                                       formatter_class=ArgumentDefaultsHelpFormatter,
                                       conflict_handler='resolve')
    embed_parser.add_argument('--input', required=True,
                              help='Input graph file')
    embed_parser.add_argument('--directed', default=True, type=bool,
                              help='input graph is directed or undirected, default = true')
    embed_parser.add_argument('--sample', nargs="*", type=str, default="none",
                              help='How to shrink the original graph, default = none')
    embed_parser.add_argument('--method', default='attention', type=str,
                              help='Embedding method (attention, 3type, sine), default = attention')
    embed_parser.add_argument('--embedtype', default='py',type=str,
                              help='Cpp or py embedding for node2vec , default = Cpp')
    
    embed_parser.add_argument('--task', default='link',type=str,
                              help='link or sign prediction , default = link')
    
    embed_parser.add_argument('--dimension', default=20, type=int,
                              help='Dimension of each node embedding, default = 10')
    embed_parser.add_argument('--walklen', default=50, type=int,
                              help='Length of walks, default = 50')
    embed_parser.add_argument('--nbofwalks', default=15, type=int,
                              help='Number of walks per source, default = 15')
    embed_parser.add_argument('--windowsize', default=5, type=int,
                              help='Window size for word2vec, default = 5')
    embed_parser.add_argument('--temp-dir', required=True, type=str,
                              help='Folder to keep intermediary files and embeddings')
    embed_parser.add_argument('--temp-id', default='', type = str,
                              help='An identifier as a prefix for temporary files')
    embed_parser.add_argument('--logfile', default='log.txt', type=str,
                              help='File name for models logs, defualt=log.txt')
    embed_parser.add_argument('--train-ratio', default=0.8, type=float,
                              help='Ratio of edges kept for training, default = 0.8')
    embed_parser.add_argument('--classificationfunc', default='Logistic', type=str,
                              help='Classification function , default = Logistic')

    embed_parser.add_argument('--optimizeclassifier', default=False, type=bool,
                              help='Classification function , default = Logistic')

    embed_parser.add_argument('--force', nargs="*", type=str, default="none",
                              help='Force redoing sample, preprocess, embed, '
                                   'postprocess, learn, default = none')

    # ------ Predict
    predict_parser = commands.add_parser('predict',
                                         formatter_class=ArgumentDefaultsHelpFormatter,
                                         conflict_handler='resolve')
    predict_parser.add_argument('--input', required=True,
                                help='a query list composed of node pairs')
    predict_parser.add_argument('--output', required=True,
                                help='result of sign prediction')
    predict_parser.add_argument('--temp-dir', required=True,
                                help='Folder of embeddings and graphs')
    predict_parser.add_argument('--temp-id', default='',
                                help='Files identifier')
    predict_parser.add_argument('--method', default='attention',
                                help='Embedding method (attention, 3type, sine)')
    args = parse_args(parser, commands)


    return args


def already_processed(processed_paths, force):
    """
        Check if re-process is not forced and all processed files exists
    :param processed_paths:
    :param force: is re-processing forced?
    :return:
    """
    if force:
        return False    # re-processing is forced for this phase
    all_processed = True
    for path in processed_paths:
        if not Path(path).is_file():
            all_processed = False
            break
    return all_processed


def force_warning(goal, phase):
    print("%s is already done, to redo add '%s' to --force" % (goal, phase))


def sample(args):
    sample_path = util.change_path(path=args.temp_dir, pre=args.temp_id,
                                   post="-sampled", ext="txt")
    PHASE = "sample"
    if already_processed([sample_path], force=PHASE in args.force):
        force_warning("sampling", PHASE)
        graph_sampled = pre.graph_load(sample_path)
    elif args.sample[0] == "degree":
        min_degree = int(args.sample[1])
        graph = pre.graph_load(args.input, is_directed=args.directed)
        node_count, edge_count, _, max_id = pre.graph_statistics(graph)

        print('graph node count: %d, edge_count: %d, max_id: %d (directed: %r)'
              % (node_count, edge_count, max_id, args.directed))
        graph_sampled = pre.graph_clean(graph, min_degree=min_degree)
        pre.graph_save(graph_sampled, sample_path)
    else:
        graph_sampled = pre.graph_load(args.input, is_directed=args.directed)
        pre.graph_save(graph_sampled, sample_path)

    return graph_sampled

def preprocess(graph, args):
    method_path = util.change_path(path=args.temp_dir, pre="%s-%s" %(args.temp_id, args.task),
                                   post="-%s" % args.method, ext="txt")
    PHASE = "preprocess"
    is_forced = PHASE in args.force  # if [re-]processing this phase is forced

    # ---------------- 3 Types ------------------ #
    if args.method == "3type" or args.method == "attention":
        new2old_path = util.change_path(method_path, post="-new2old")
        old2new_path = util.change_path(method_path, post="-old2new")
        train_path = util.change_path(method_path, post="-train")
        test_path = util.change_path(method_path, post="-test")
        train_absent_path = util.change_path(method_path, post="-train-absent")
        test_absent_path = util.change_path(method_path, post="-test-absent")

        if already_processed([method_path, new2old_path, old2new_path,
                              train_path, test_path], force=is_forced):
            force_warning(args.method, PHASE)
        else:
            # transform graph into undirected and unsigned
            graph_3types, new2old, old2new = pre.to_3types_full(graph)
            node_count, edge_count, _, max_id = pre.graph_statistics(graph_3types)
            print('3types graph node count: %d, edge_count: %d, max_id: %d' % (node_count, edge_count, max_id))
            pre.graph_save(graph_3types, method_path)
            pre.json_save(new2old, new2old_path)
            pre.json_save(old2new, old2new_path)
            print('3types graph is saved.')
            # split graph into train and test sub-graphs
            train, test = pre.graph_split(train_ratio=args.train_ratio, graph=graph_3types, new2old=new2old, old2new=old2new)
            _, _, train_absent, test_absent = pre.graph_split_general(train_ratio=args.train_ratio, graph=graph, task=args.task)

            node_count, edge_count, _, max_id = pre.graph_statistics(train)
            print('3types train node count: %d, edge_count: %d, max_id: %d' % (node_count, edge_count, max_id))
            node_count, edge_count, _, max_id = pre.graph_statistics(test)
            print('3types test node count: %d, edge_count: %d, max_id: %d' % (node_count, edge_count, max_id))
            pre.graph_save(train, train_path)
            pre.graph_save(test, test_path)
            print('3types graph train-test split is saved.')
            if( args.task == 'link'):
                pre.graph_save(train_absent, train_absent_path)
                pre.graph_save(test_absent, test_absent_path)

    # ---------------- Unsigned ------------------ #
    if args.method == "attention":
        unsigned_path = util.change_path(method_path, post="-unsigned")
        train_path = util.change_path(unsigned_path, post="-train")
        test_path = util.change_path(unsigned_path, post="-test")
        if already_processed([method_path, train_path, test_path],
                             force=is_forced):
            force_warning(args.method + " unsigned", PHASE)
        else:
            # transform graph into undirected and unsigned 2nd order matrix
            graph_unsigned = pre.apply_on_weight(graph, lambda x: abs(x))
            node_count, edge_count, _, max_id = pre.graph_statistics(graph_unsigned)
            print('unsigned graph node count: %d, edge_count: %d, max_id: %d' % (node_count, edge_count, max_id))
            pre.graph_save(graph_unsigned, unsigned_path)
            # Split the unsigned graph into train and test
            # based on the exact split of 3 types graph
            train_3types = pre.graph_load(path=util.change_path(method_path, post="-train"))
            test_3types = pre.graph_load(path=util.change_path(method_path, post="-test"))
            new2old = pre.json_load(path=util.change_path(method_path, post="-new2old"))
            train_unsigned = pre.graph_select(graph_unsigned, pre.three_type_to_graph(train_3types, new2old))
            test_unsigned = pre.graph_select(graph_unsigned, pre.three_type_to_graph(test_3types, new2old))
            node_count, edge_count, _, max_id = pre.graph_statistics(train_unsigned)
            print('unsigned train node count: %d, edge_count: %d, max_id: %d' % (node_count, edge_count, max_id))
            node_count, edge_count, _, max_id = pre.graph_statistics(test_unsigned)
            print('unsigned test node count: %d, edge_count: %d, max_id: %d' % (node_count, edge_count, max_id))
            pre.graph_save(train_unsigned, train_path)
            pre.graph_save(test_unsigned, test_path)
            print('unsigned graph train/test split is saved.')



def embed(args, max_node_id, sine_epochs=30):
    """
        Embed nodes of the sampled graph and same them into temp folder
    :param args:
    :param max_node_id: SINE method needs the node count as input
    :param sine_epochs: epochs for sine embedding outer loop
    :return:
    """
    PHASE = "embed"
    is_forced = PHASE in args.force     # if [re-]processing this phase is forced
    method_path = util.change_path(path=args.temp_dir, pre="%s-%s" %(args.temp_id, args.task),
                                   post="-%s" % args.method, ext="txt")
    if os.name == 'nt':
        is_windows = True
        if args.method == "3type" or args.method == "attention":
            print('OS is windows, for node2vec.cpp, '
                  '%s and %s must be set in config.ini' % (wiki.const.SHELL, wiki.const.NODE_TO_VEC))
    else:
        is_windows = False

    if args.method == "3type" or args.method == "attention":
        graph_path = util.change_path(path=method_path, post="-train")
        output_path = util.change_path(path=method_path, post="-embed-raw")
        if already_processed([output_path], force=is_forced):
            force_warning(args.method + " " + PHASE, PHASE)
        else:
            if args.embedtype == "Cpp":
                mynode2vec.embed_cpp(graph_path=graph_path, output_path=output_path,
                               dimension=args.dimension, walk_length= args.walklen, nb_of_walk_per_srouce= args.nbofwalks,
                               is_directed=False, is_windows=is_windows)
            if args.embedtype == "py":
                mynode2vec.embed_py(graph_path=graph_path, output_path=output_path,
                               dimension=args.dimension, walk_length= args.walklen, nb_of_walk_per_srouce= args.nbofwalks,
                               is_directed=False, is_windows=is_windows, window_size= args.windowsize)

    if args.method == "attention":
        graph_path = util.change_path(path=method_path, post="-unsigned-train")
        output_path = util.change_path(path=method_path, post="-unsigned-embed-raw")
        if already_processed([output_path], force=is_forced):
            force_warning(args.method + " unsigned " + PHASE, PHASE)
        else:
            if args.embedtype == "cpp":
                mynode2vec.embed_cpp(graph_path=graph_path, output_path=output_path,
                               dimension=args.dimension, walk_length= args.walklen, nb_of_walk_per_srouce= args.nbofwalks,
                               is_directed=False, is_windows=is_windows)
            if args.embedtype == "py":
                mynode2vec.embed_py(graph_path=graph_path, output_path=output_path,
                               dimension=args.dimension, walk_length= args.walklen, nb_of_walk_per_srouce= args.nbofwalks,
                               is_directed=False, is_windows=is_windows, window_size= args.windowsize)


def postprocess(args, neg2pos_ratio=1.0, sine_model="30.p"):
    """

    :param args:
    :param neg2pos_ratio: ratio of negative edges to positive ones to balance the training set
    :param sine_model: sine model to load embeddings from
    :return:
    """
    PHASE = "postprocess"
    is_forced = PHASE in args.force  # if [re-]processing this phase is forced
    similarity_type = "distance"  # distance product
    paths = sign_prediction_paths(args)

    # ----------------- Node2Vec on 3 Types ----------------- #
    if args.method == "3type":
        if already_processed([paths['embed'], paths['link_train'], paths['link_test']], force=is_forced):
            force_warning(args.method + " " + PHASE, PHASE)
        else:
            post.n2v_save_embeddings_to_mat(embed_path=paths['embed_raw'], save_path=paths['embed'])
            # create train dataset from embeddings
            print('Train:')
            graph_train = pre.graph_load(paths['train'])
            graph_absent_train = {}
            graph_absent_test = {}
            if args.task == 'link':
                graph_absent_train = pre.graph_load(paths['train-absent'])
                graph_absent_test = pre.graph_load(paths['test-absent'])


            post.generate_dataset_3type(graph=graph_train, graph_absent=graph_absent_train, embed_path=paths['embed'], new2old_path=paths['new2old'],
                                        old2new_path=paths['old2new'],
                                        neg2pos_ratio=neg2pos_ratio,
                                        save_path=paths['link_train'], task= args.task)
            # create test dataset from embeddings
            print('Test:')
            graph_test = pre.graph_load(paths['test'])
            post.generate_dataset_3type(graph=graph_test, graph_absent=graph_absent_test, embed_path=paths['embed'], new2old_path=paths['new2old'],
                                        old2new_path=paths['old2new'],
                                        neg2pos_ratio=neg2pos_ratio,
                                        save_path=paths['link_test'], task= args.task)
    # ----------------- Node2Vec on Attention ----------------- #
    if args.method == "attention":
        if already_processed([paths['embed'], paths['unsigned_embed'],
                              paths['link_train'], paths['link_test']], force=is_forced):
            force_warning(args.method + " " + PHASE, PHASE)
        else:
            post.n2v_save_embeddings_to_mat(embed_path=paths['embed_raw'],
                                            save_path=paths['embed'])
            post.n2v_save_embeddings_to_mat(embed_path=paths['unsigned_embed_raw'],
                                            save_path=paths['unsigned_embed'])
            # create train dataset from embeddings
            print('Train:')
            graph_train = pre.graph_load(paths['train'])
            graph_absent_train = {}
            graph_absent_test = {}
            if args.task == 'link':
                graph_absent_train = pre.graph_load(paths['train-absent'])
                graph_absent_test = pre.graph_load(paths['test-absent'])

            post.generate_dataset_3type(graph=graph_train, graph_absent=graph_absent_train, embed_path=paths['embed'],
                                        unsigned_embed_path=paths['unsigned_embed'],
                                        relations_path=paths['train'],
                                        new2old_path=paths['new2old'], old2new_path=paths['old2new'],
                                        similarity_type=similarity_type,
                                        neg2pos_ratio=neg2pos_ratio,
                                        save_path=paths['link_train'], task= args.task)
            # create test dataset from embeddings
            # whole (test + train) graph can be used for relations
            print('Test:')
            graph_test = pre.graph_load(paths['test'])
            post.generate_dataset_3type(graph=graph_test, graph_absent=graph_absent_test,  embed_path=paths['embed'],
                                        unsigned_embed_path=paths['unsigned_embed'],
                                        relations_path=paths['graph'],
                                        new2old_path=paths['new2old'], old2new_path=paths['old2new'],
                                        similarity_type=similarity_type,
                                        neg2pos_ratio=neg2pos_ratio,
                                        save_path=paths['link_test'], task= args.task)


def sign_prediction_learn(args):
    PHASE = "model"
    paths = sign_prediction_paths(args)
    if already_processed([paths['link_model']], force=PHASE in args.force):
        force_warning(args.method + " sign prediction model", PHASE)
    else:
        sign_prediction.train(dataset_path=paths['link_train'], save_path=paths['link_model'],
                              classification_model = args.classificationfunc, search_parameters = args.optimizeclassifier, args=args)
    # on train data
    print('Train:')
    sign_prediction.test(dataset_path=paths['link_train'], model_path=paths['link_model'], args=args)
    # on test data
    print('Test:')
    sign_prediction.test(dataset_path=paths['link_test'], model_path=paths['link_model'], args=args)



def sign_prediction_predict(args):
    similarity_type = "distance"  # distance product
    paths = sign_prediction_paths(args)
    graph = pre.to_weighted(pre.graph_load(args.input), weight=1)
    model = sign_prediction.model_load(paths['link_model'])
    dataset = None
    if args.method == "sine":
        dataset = post.generate_dataset_sine(graph=graph, embed_path=paths['embed'])
    elif args.method == "3type" or args.method == "attention":
        # convert the query graph to 3types graph based on id maps of original graph
        graph_3type, _, _ = pre.to_3types_full(graph=graph, old2new=pre.json_load(paths['old2new']))
        if args.method == "3type":
            dataset = post.generate_dataset_3type(graph=graph_3type, embed_path=paths['embed'],
                                                  new2old_path=paths['new2old'],
                                                  old2new_path=paths['old2new'])
        elif args.method == "attention":
            dataset = post.generate_dataset_3type(graph=graph_3type, embed_path=paths['embed'],
                                                  unsigned_embed_path=paths['unsigned_embed'],
                                                  relations_path=paths['train'],
                                                  new2old_path=paths['new2old'],
                                                  old2new_path=paths['old2new'])
    else:
        raise Exception("Method %s not supported" % args.method)
    prediction_scores = sign_prediction.predict(dataset, model, mode="score")
    source_ids = dataset['metadata'][:, 0]
    target_ids = dataset['metadata'][:, 1]
    output = np.vstack([source_ids, target_ids, prediction_scores]).transpose()
    np.savetxt(args.output, output, fmt=['%d', '%d', '%.2f'], delimiter=' ')


def sign_prediction_paths(args):
    paths = {'graph': util.change_path(path=args.temp_dir, pre="%s-%s" %(args.temp_id, args.task),
                                       post="-%s" % args.method, ext="txt")}
    paths['train'] = util.change_path(path=paths['graph'], post="-train")
    paths['test'] = util.change_path(path=paths['graph'], post="-test")
    paths['train-absent'] = util.change_path(path=paths['graph'], post="-train-absent")
    paths['test-absent'] = util.change_path(path=paths['graph'], post="-test-absent")

    paths['new2old'] = util.change_path(path=paths['graph'], post="-new2old")
    paths['old2new'] = util.change_path(path=paths['graph'], post="-old2new")
    paths['embed'] = util.change_path(path=paths['graph'], post="-embed", ext="mat")
    paths['embed_raw'] = util.change_path(path=paths['graph'], post="-embed-raw")
    paths['unsigned_embed'] = util.change_path(path=paths['graph'], post="-unsigned-embed", ext="mat")
    paths['unsigned_embed_raw'] = util.change_path(path=paths['graph'], post="-unsigned-embed-raw")
    paths['link_train'] = util.change_path(path=paths['graph'], post="-link-train", ext="mat")
    paths['link_test'] = util.change_path(path=paths['graph'], post="-link-test", ext="mat")
    paths['link_model'] = util.change_path(path=paths['graph'], post="-link-model", ext="mdl")
    return paths

def models_dir(args):
    return "%s//%s-%s-%s-models//" % (args.temp_dir, args.temp_id, args.task, args.method)


def main(args):
    
    if args.build is not None:

        print('--------- PreProcess -----------')
        args = args.build

        graph_sampled = sample(args)
        node_count, edge_count, negative_edge_count, max_id = pre.graph_statistics(graph_sampled)
        negative_edge_ratio = negative_edge_count / edge_count
        neg2pos_ratio = negative_edge_ratio / (1.0001 - negative_edge_ratio)
        print('sampled graph node count: %d, edge_count: %d, negative_ratio: %.2f, max_id: %d'
              % (node_count, edge_count, negative_edge_ratio, max_id))
        preprocess(graph_sampled, args)
        # embed the sampled graph
        print('---------    Embed    -----------')
        embed(args=args, max_node_id=max_id, sine_epochs=30)
        # post-process the embeddings into txt files
        # and prepare train-test sets for sign prediction
        print('--------- PostProcess -----------')
        postprocess(args, neg2pos_ratio=neg2pos_ratio, sine_model="30.p")
        # learn the model
        print('--------- Learn Model -----------')
        sign_prediction_learn(args)
    elif args.predict is not None:
        args = args.predict
        sign_prediction_predict(args)
    return True


if __name__ == '__main__':
    args = get_args()
    main(args)
