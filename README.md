#### Setup

1. Download and make node2vec from
[Github](https://github.com/snap-stanford/snap/tree/master/examples/node2vec),
1. Copy `default.config.ini` to `config.ini`,
1. Set the node2vec executable path in `config.ini > NODE_TO_VEC`,
1. In windows, set `sh.exe` path in `config.ini > SHELL`

Option 1: 
#### Use Script


Execute ./test_script $1 $2 $3 $4 
    1. $1: dataset [wiki|slashdot|epinions]
    2. $2: method of embedding [3type|attention] 
    3. $3: node degree threshold for sampling from the input graph (shrinks the graph by removing nodes with degree < k). Set to 0 if you want to use the entire graph. 
    4. $4: target task (the model can be used for link prediction and sign prediction) [link|sign]

Option 2:
#### Run from Command line


1. To embed a graph and build a sign prediction model, execute:
    ```
    src/bootstrap.py build --input slashdot.txt --temp-dir temp/ --temp-id slash
    ```
    **Note**: do not forget `\` or `/` at the end of `--temp-dir`,
    1. Temporary files will be saved in `temp/slash-*`,
    1. `--method [3type | attention | sine]` sets the method of embedding,
    1. `--dimension 20` sets the dimension of each node embedding,
    1. `--train-ratio 0.8` keeps 20% of links for testing,
    1. `--sample_mode clean 4` shrinks the graph by removing nodes with degree < 4,
    1. `--force [sample] [preprocess] [embed] [postprocess] [model]` forces the re-doing of phases in the list, otherwise their previous temporary files will be used
2. To predict signs for a query file, containing `node1 node2 [weight]` per line, execute:
    ```
    src/bootstrap.py predict --input query.txt --output scores.txt --temp-dir temp/ --temp-id slash
    ```
    1. Model and embeddings will be read from `temp/slash-*`,
    1. `--method [3type | attention | sine]` sets the method of embedding


Option 3:

1. See `src/bootstrap_run.py` that creates an argument object,
    and passes it to main function of `src/bootstrap.py`
