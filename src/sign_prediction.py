
from scipy import io
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np

def model_load(path):
    return joblib.load(path)

def decompose(dataset, task='link'):
    """"
    :param dataset:
    :return: (features, label, metadata)
    """
    if task == 'link':
        dataset['dataset'][:, -1] = list(map(lambda x: abs(x), dataset['dataset'][:, -1]))


    return dataset['dataset'][:, :-1], \
           dataset['dataset'][:, -1], \
           dataset['metadata'][:, :]

def train(dataset_path, save_path, classification_model = "Logistic", search_parameters = False, args=None):
    dataset_all = io.loadmat(dataset_path)
    features, label, meta = decompose(dataset_all, args.task)
    print('Feature count is %d' % features.shape[1])

    param_grid = [
        {
	    'activation': ['relu'],
            'solver' : ['adam'],
            'hidden_layer_sizes': [
                (7,3)
             ]
        }
       ]
    if search_parameters == True:
        param_grid = [
            {
                'activation' : [  'relu'],
                'solver' : [ 'adam'],
                'hidden_layer_sizes': [
                     (9,3) #, (6,2), (6)#,(16,8),(8,4),(8,2)
                 ]
            }
           ]
    model = LogisticRegression(random_state=0, solver='lbfgs')
    if classification_model == "Logistic":
        model = model.fit(features, label)

    if classification_model == "MLP":
        model = GridSearchCV(MLPClassifier(random_state=1,learning_rate='adaptive', alpha=0.05, early_stopping=True,max_iter=500, validation_fraction=0.07), param_grid, cv=3, scoring='accuracy')
        model.fit(features, label)
        #model = MLPClassifier(solver='sgd', alpha=1e-6,
        #            hidden_layer_sizes=(7,2), random_state=1).fit(features, label)
        print("Best parameters set found on development set:")
        logfile = open(args.temp_dir+args.logfile, "a")
        logfile.write(args.temp_id+"\n")
        logfile.write("Best parameters set found on development set:"+"\n")
        logfile.write(str(model.best_params_)+"\n")
        logfile.write("------------------ \n")
        logfile.close()
        print(model.best_params_)

    # save model
    joblib.dump(model, save_path)

def test(dataset_path, model_path, args):
    logfile = open(args.temp_dir+args.logfile, "a")
    logfile.write(model_path+"\n")
    logfile.write(args.temp_id+"\n")
    dataset_all = io.loadmat(dataset_path)
    model = model_load(model_path)
    _, label, _ = decompose(dataset_all,args.task)
    # measure classifier performance on test set
    predicted_label = predict(dataset_all, model, mode="label", task=args.task)
    f1_micro = sklearn.metrics.f1_score(label, predicted_label, pos_label=1, average='micro')
    f1_macro = sklearn.metrics.f1_score(label, predicted_label, pos_label=1, average='macro')
    logfile.write('Micro f1: %.2f%%, macro f1: %.2f%% \n' % (f1_micro * 100, f1_macro * 100))
    print('Micro f1: %.2f%%, macro f1: %.2f%%' % (f1_micro * 100, f1_macro * 100))
    predicted_degree = predict(dataset_all, model, mode="score", task=args.task)
    fpr, tpr, thresholds = metrics.roc_curve(label, predicted_degree, pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
    logfile.write('AUC: %.2f%% \n' % (auc * 100))
    print('AUC: %.2f%%' % (auc * 100))
    logfile.write('----------------------------------- \n')
    logfile.close()

def predict(dataset, model, mode, task):
    features, _, meta = decompose(dataset, task)
    if mode == "label":
        output = model.predict(features)
    elif mode == "score":
        # predict_probability returns (negative prob, positive prob)
        # where positive prob + negative prob = 1
        probabilities = model.predict_proba(features)
        output = 2 * probabilities[:, 1] - 1
    else:
        raise Exception("Prediction mode %s is not supported" % mode)
    # has_target_pos = meta[:, 2] if meta.shape[1] > 2 else np.ones(meta.shape[0])
    # has_target_neg = meta[:, 3] if meta.shape[1] > 2 else np.ones(meta.shape[0])
    # # no +1 link into target, so link = -1
    # # +1 is more common so it can overwrite -1
    # output[has_target_pos == 0] = -1 if mode == "label" else -0.5
    # output[has_target_neg == 0] = 1 if mode == "label" else 0.5
    return output

