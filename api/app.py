# This is where I can run the trained model and get predictions.
# import keras
import sys
import os
#add root dir to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import flask
from flask import Flask, jsonify
from data.dataset import TweetDataset
from networks.lstm_basic import lstm_basic
from model.model import LSTMModel
import random
from flask import render_template




# import predictor

model = None
seed_one_hot = None

# path = '../data/tumpstweets_source.txt'
# model_file = '../model_data/lstm-big-brain-model_mxnet.json'
# weights_file = '../model_data/lstm-big-brain_new_mxnet.h5'

def load_model():
    global model
    global seed_one_hot
    path = "../data/nietzsche.txt"
    params = "../lstm_basic.params"
    #load dataset
    data = TweetDataset(path, data_size=1)
    data.create_sequances()
    X, Y = data.vectorize_all_sequances()
    seed_one_hot = X[1000]
    model = LSTMModel(net=lstm_basic, output_neurons=len(data.chars))
    #load saved params to the model
    model.load_params(params)
    model.net_info()
    #load corpus
    model.load_corpus(data.char_to_ind, data.ind_to_char, data.chars)



    
# init flask
app = Flask(__name__)

# pred = predictor.TweetPredictor(model_file, weights_file, path)


@app.route('/')
def index():
    return render_template("one-tweet.html")

@app.route("/predict")
def predict():
    seed, generated = model.predict_text(seed_one_hot)
    # data["seed"] = seed
    # data["text"] = generated
    
    #     # initialize the data dictionary that will be returned from the
    # # view
    # data = {"success": False}
    
    # # ensure an image was properly uploaded to our endpoint
    # if flask.request.method == "POST":
    #     if flask.request.get_json():
    #         seed = flask.request.get_json()['seed']
    #         print(seed)
    #         # global graph
    #         # with graph.as_default():
    #         seed, generated = model.predict_text(seed_one_hot)
    #         data["seed"] = seed
    #         data["text"] = generated
    #         # indicate that the request was a success
    #         data["success"] = True

    # return json data back to client
    return render_template("one-tweet.html", text=generated)
    


if __name__ == "__main__":
    print("[INFO] Loading LSTM model and starting Flask API endpoint.")
    load_model()
    app.run(host='0.0.0.0')