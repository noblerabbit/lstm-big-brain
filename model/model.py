"""
This file here is loaed to init the model to for training and also for api inference
Therefore the training code is not part of this model, only model initialization (net, net params)
and inference methods.
"""
from networks.lstm_basic import lstm_basic
import mxnet as mx
import numpy as np
from mxnet import nd
# from data.dataset import TweetDataset


class LSTMModel():
    def __init__(self, net=lstm_basic, output_neurons = 195, ctx=mx.gpu(0)):
        self.ctx=ctx
        self.net = net(output_neurons)
        self.char_to_ind = {}
        self.ind_to_char = {}
        self.chars = []
        self.corpus_loaded = False
    
    def net_info(self):
        print(self.net)
        
    def initilize(self):
        self.net.initialize(ctx=ctx)
        
    def load_params(self, paramsfname):
        self.net.load_parameters(paramsfname, ctx=self.ctx)
    
    def save_params(self, paramsfname):
        self.net.save_parameters(paramsfname)

    def load_corpus(self, char_to_ind, ind_to_char, chars):
        self.char_to_ind = char_to_ind
        self.ind_to_char = ind_to_char
        self.chars = chars
        self.corpus_loaded = True
        
    def predict_text(self, seed_one_hot, length=160):
        if not self.corpus_loaded:
            print("[ERROR] Corpus not Loaded. Run load_corpus(*args) first.")
            return -1
        seed = seed_one_hot
        text = ''
        for i in range(length):
            output = self.net(nd.array(seed, ctx=self.ctx).expand_dims(axis=0)).softmax()
            indice = np.argmax(output[0].asnumpy())
            text = text + self.ind_to_char[indice]
            char_vec = np.zeros(len(output[0]))
            char_vec[indice] = 1
            seed = np.vstack((seed, char_vec))[1:]
        print(self.to_text(seed_one_hot),": ", text)
        
    def to_text(self, one_hot_sentance):
        text = ''
        for char in one_hot_sentance:
            text = text + self.ind_to_char[np.argmax(char)]
        return text
                
        
        