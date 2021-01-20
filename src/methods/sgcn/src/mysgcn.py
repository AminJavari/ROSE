from src.methods.sgcn.src.sgcn import SignedGCNTrainer
from src.methods.sgcn.src.utils import tab_printer, read_graph, score_printer, save_logs
from collections import namedtuple

def sgcn(iedge_path, imodel_path):
    """
    Parsing command line parameters, creating target matrix, fitting an SGCN, predicting edge signs, and saving the embedding.
    """
    buildClass = namedtuple('buildClass', 'edge_path model_path features_path embedding_path regression_weights_path log_path epochs '
                                          'reduction_iterations reduction_dimensions seed lamb test_size learning_rate weight_decay layers '
                                          'spectral_features ')
    args = buildClass(edge_path=iedge_path, model_path=imodel_path, features_path=' ', embedding_path=imodel_path+'emb.csv',
                      regression_weights_path=imodel_path+'weights.csv', epochs=100, reduction_iterations = 30,  reduction_dimensions =5,
                    seed=42, lamb=1.0, test_size=0.2, learning_rate=0.01, weight_decay=10**-5, layers=[32,32], spectral_features = True,
                      log_path = imodel_path+'logs.json')

    tab_printer(args)
    edges = read_graph(args)
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)


