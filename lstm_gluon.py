from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)


path = 'data/Tumpstweets_source.txt'

with open(path) as f:
    time_machine = f.read()
time_machine = time_machine[:1000000]


character_list = list(set(time_machine))
vocab_size = len(character_list)
character_dict = {}
for e, char in enumerate(character_list):
    character_dict[char] = e
time_numerical = [character_dict[char] for char in time_machine]

def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result