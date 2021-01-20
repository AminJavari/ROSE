import pandas as pd
import numpy as np
import os

def prone_vocab(vocab: pd.DataFrame):
    del vocab[1]
    del vocab[2]
    vocab = vocab.replace({'b': ''}, regex=True)

    vocab= vocab.rename(columns={0: 'id'})
    for index, row in vocab.iterrows():
        vocab['id'][index] = vocab['id'][index][1:-1]

    vocab['id']= vocab['id'].astype(int)
    return vocab

def df2array_vocab(df: pd.DataFrame, vocab: pd.DataFrame):
    df = pd.concat([vocab, df], axis=1, sort=False)
    max_id = max(vocab['id'])
    array = np.zeros(shape=(max_id + 1, len(df.columns) - 1))

    for _, record in df.iterrows():
        row = record.values
        array[int(row[0]), :] = row[1:]
    return array


def df2array(df: pd.DataFrame):
    """
        Index of row is supposed to be in column "0"
    :param df:
    :return: array
    :rtype:
    """
    new_df = df.sort_values(by=df.columns[0])
    max_id = new_df[0].max()
    array = np.zeros(shape=(int(max_id + 1), len(df.columns) - 1))
    for _, record in df.iterrows():
        row = record.values
        array[int(row[0]), :] = row[1:]
    return array


def change_path(path, dir="", file="", pre="", post="", ext=""):
    """
        Change the path ingredients with the provided directory, filename
        prefix, postfix, and extension
    :param path:
    :param dir: new directory
    :param file: filename to replace the filename full_path
    :param pre: prefix to be appended to filename full_path
    :param post: postfix to be appended to filename full_path
    :param ext: extension of filename to be changed
    :return:
    """

    from pathlib import Path
    target = ""
    path_obj = Path(path)
    
    old_filename = path_obj.name.replace(path_obj.suffix, "") \
        if len(path_obj.suffix) > 0 else path_obj.name

    if os.name == "nt":
        if len(dir) > 0:
            directory = dir
        elif path.endswith("\\"):
            directory = path[:-1]
            old_filename = ""
        else:
            directory = str(path_obj.parent)
        old_extension = path_obj.suffix
        new_filename = file if len(file) > 0 else old_filename
        new_filename = pre + new_filename if len(pre) > 0 else new_filename
        new_filename = new_filename + post if len(post) > 0 else new_filename
        new_extension = "." + ext if len(ext) > 0 else old_extension
        target = directory + "\\" + new_filename + new_extension
    else:

        if len(dir) > 0:
            directory = dir
        elif path.endswith("/"):
            directory = path[:-1]
            old_filename = ""
        else:
            directory = str(path_obj.parent)
        
        
	
        old_extension = path_obj.suffix
        new_filename = file if len(file) > 0 else old_filename
        new_filename = pre + new_filename if len(pre) > 0 else new_filename
        new_filename = new_filename + post if len(post) > 0 else new_filename
        
        new_extension = "." + ext if len(ext) > 0 else old_extension
        target = directory + "/" + new_filename + new_extension
        

    return target
