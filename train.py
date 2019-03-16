from data.dataset import TweetDataset
from networks.lstm_basic import lstm_basic
from mxnet import nd, init, gluon, autograd
import mxnet as mx
import time
import numpy as np

#path to source file
path = "data/tumpstweets_source.txt"
params = "lstm_basic.params"

#load dataset
data = TweetDataset(path, data_size=0.01)
data.create_sequances()
X, Y = data.vectorize_sequances()

#load model
net = lstm_basic(len(data.chars))

#init the model
net.initialize(ctx=mx.gpu(0))
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
trainer = gluon.Trainer(net.collect_params(), 'RMSProp', {'learning_rate':0.01})


batch_size = 2048
for epoch in range(4):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    for i in range(0, X.shape[0]-batch_size, batch_size):
        x_batch = nd.array(X[i: i + batch_size,:,:], ctx=mx.gpu(0))
        y_batch = nd.array(Y[i: i + batch_size,:], ctx=mx.gpu(0))
        with autograd.record():
            output = net(x_batch)
            loss = softmax_cross_entropy(output, y_batch)
        loss.backward()
        trainer.step(batch_size)
        train_loss += loss.mean().asscalar()
#         print(train_loss)
    print("Epoch {}: loss: {}, time: {} seconds.".format(epoch, train_loss, time.time()-tic))
    
net.save_parameters(params)

def to_text(one_hot_sentance):
    text = ''
    for char in one_hot_sentance:
        text = text + data.ind_to_char[np.argmax(char)]
    return text
    
def predict_text(one_hot_sentance):
    seed=one_hot_sentance
    text = ''
    for i in range(160):
        output = net(nd.array(seed, ctx=mx.gpu(0)).expand_dims(axis=0)).softmax()
        indice = np.argmax(output[0].asnumpy())
        text = text + data.ind_to_char[indice]
        char_vec = np.zeros(len(output[0]))
        char_vec[indice] = 1
        seed = np.vstack((seed, char_vec))[1:]
    print(to_text(one_hot_sentance), text)
        
indice = predict_text(X[1000])