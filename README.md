# Applying CNNs to Sentence Generation

This project uses a DenseNet for character and word level language modelling. Models can be toggled with the mode setting set to 'char' or 'word' and should be created in a `models/` directory. Data should be loaded into the `data/` folder.

See `run.py` for details on the training and testing loops. 

See `densenet.py` for details on the model architectures.

See `script.py` for a scratch pad for code testing.

1) Run `./download.sh` to download the 10GB files required for fasttext embeddings.
2) Then run `python run.py` with paths to your data files setup.

For help, look at the argument parser to see what is required and what sensible defaults are. This work was tested on the English Penn Treebank (download here: https://github.com/yoonkim/lstm-char-cnn/tree/master/data/ptb) and the 1 billion word dataset (download here: http://www.statmt.org/lm-benchmark/). You should put these files into the `data/` folder to use them
