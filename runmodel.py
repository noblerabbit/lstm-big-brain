# This is where I can run the trained model and get predictions.
# import keras
from keras.models import model_from_json
import numpy as np
import sys
import io
import random
import flask
import tensorflow as tf
import time

path = 'data/Tumpstweets_source.txt'
model_file = 'model_data/lstm-big-brain-model.json'
weights_file = 'model_data/lstm-big-brain.h5'

loaded_model = None #placce holder var for model
graph = None

with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
    text = text[:1000000] # to fit my gpu
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

###

def load_model():
    global loaded_model
    global graph
    graph = tf.get_default_graph()
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Loaded model from json file")
    # print(loaded_model.summary())

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def pred_on_seed():
    tweets=[]
    
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        for i in range(160):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = loaded_model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        tweets.append(generated)
            
    return tweets

# init flask
app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
        # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json():
            seed = flask.request.get_json()['seed']
            print(seed)
            
            with graph.as_default():
                tweet = pred_on_seed()
            data["tweet"] = tweet
            # indicate that the request was a success
            data["success"] = True

    # return json data back to client
    return flask.jsonify(data)
    


if __name__ == "__main__":
    print("[INFO] Loading LSTM model and starting Flask API endpoint.")
    load_model()
    app.run()
