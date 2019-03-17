from data.dataset import TweetDataset
from networks.lstm_basic import lstm_basic
from mxnet import nd, init, gluon, autograd
import mxnet as mx
import time
import numpy as np
from model.model import LSTMModel
import random

#path to source file
path = "data/nietzsche.txt"
params = "lstm_basic.params"

BATCH_SIZE = 4096
EPOCHS = 100

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
chunk_size = int(len(data.text)*1) # we load as much as we can in RAM
STEPS_PER_EPOCH = int(len(data.text)/chunk_size)
print("STEPS_PER_EPOCH : {}".format(STEPS_PER_EPOCH))

X, Y = data.vectorize_all_sequances() # if dataset is small enough to fit in RAM

for epoch in range(EPOCHS):
    for step in range(STEPS_PER_EPOCH):
        # X, Y = next(data.get_batch(chunk_size))
        # print(X.shape)
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()
        for i in range(0, X.shape[0]-BATCH_SIZE, BATCH_SIZE):
            # print(i)
            x_batch = nd.array(X[i: i + BATCH_SIZE,:,:], ctx=mx.gpu(0))
            y_batch = nd.array(Y[i: i + BATCH_SIZE,:], ctx=mx.gpu(0))
            with autograd.record():
                output = model.net(x_batch)
                loss = softmax_cross_entropy(output, y_batch)
            loss.backward()
            trainer.step(BATCH_SIZE)
            train_loss += loss.mean().asscalar()
        print("[INFO] Step {}: loss: {}, time: {} seconds.".format(step, train_loss/BATCH_SIZE, time.time()-tic))
        print("[INFO] Predicting the text:")
        # model.predict_text(X[random.randint(0, len(X))])
    #         print(train_loss)
    print("[INFO] Epoch {}: loss: {}, time: {} seconds.".format(epoch, train_loss, time.time()-tic))
    model.predict_text(X[random.randint(0, len(X))])

    
    
model.net.save_parameters(params)


#predict text
# model.predict_text(X[1000])
