from data.dataset import TweetDataset
from networks.lstm_basic import lstm_basic
from mxnet import nd, init, gluon, autograd
import mxnet as mx
import time
import numpy as np
from model.model import LSTMModel

#path to source file
# path = "data/tumpstweets_source.txt"
path = "data/nietzsche.txt"
params = "lstm_basic.params"

#load dataset
data = TweetDataset(path, data_size=1)
data.create_sequances()
X, Y = data.vectorize_all_sequances()


#load model
model = LSTMModel(net=lstm_basic, output_neurons=len(data.chars))

#load saved params to the model
model.load_params(params)
model.net_info()

#load corpus
model.load_corpus(data.char_to_ind, data.ind_to_char, data.chars)

#predict text
print(model.to_text(X[1000]))
model.predict_text(X[1000])