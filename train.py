from data.dataset import TweetDataset
from networks.lstm_basic import lstm_basic
from mxnet import nd, init, gluon, autograd
import mxnet as mx
import time
import numpy as np
from model.model import LSTMModel

#path to source file
path = "data/tumpstweets_source.txt"
params = "lstm_basic.params"

BATCH_SIZE = 2048
EPOCHS = 4

#load dataset
data = TweetDataset(path, data_size=1)
data.create_sequances()
# X, Y = data.vectorize_sequances()


#load model
model = LSTMModel(net=lstm_basic, output_neurons=len(data.chars))

#load corpus for prediction
model.load_corpus(data.char_to_ind, data.ind_to_char, data.chars)

#load saved params to the model
# model.load_params(params)
# or init empy model
model.initilize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
trainer = gluon.Trainer(model.net.collect_params(), 'RMSProp', {'learning_rate':0.01})

#model summary
model.net_info()

# train the model
for epoch in range(EPOCHS):
    while True:
        try:
            X, Y = next(data.get_batch(int(len(data.text)*0.01)))
        except:
            print("Generator Ehusated. Going to next epoch.")
            break
        print(X.shape)
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()
        for i in range(0, X.shape[0]-BATCH_SIZE, BATCH_SIZE):
            print(i)
            x_batch = nd.array(X[i: i + BATCH_SIZE,:,:], ctx=mx.gpu(0))
            y_batch = nd.array(Y[i: i + BATCH_SIZE,:], ctx=mx.gpu(0))
            with autograd.record():
                output = model.net(x_batch)
                loss = softmax_cross_entropy(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)
            train_loss += loss.mean().asscalar()
    #         print(train_loss)
    print("Epoch {}: loss: {}, time: {} seconds.".format(epoch, train_loss, time.time()-tic))
    
# net.save_parameters(params)


#predict text
model.predict_text(X[1000])
