# This is where I can run the trained model and get predictions.
# import keras
from keras.models import model_from_json
from keras import backend
import tensorflow as tf

import numpy as np
import sys
import io
import random
import flask

import time
import predictor

# global graph,model
# graph = tf.get_default_graph()


path = '../data/Tumpstweets_source.txt'
model_file = '../model_data/lstm-big-brain-model.json'
weights_file = '../model_data/lstm-big-brain.h5'

# init flask
app = flask.Flask(__name__)

pred = predictor.TweetPredictor(model_file, weights_file, path)
graph = tf.get_default_graph()


# with backend.get_session().graph.as_default() as g:

@app.route('/')
def index():
    return 'Hello, world!'

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
            global graph
            with graph.as_default():
                tweet = pred.predict_tweet()
                data["tweet"] = tweet
                # indicate that the request was a success
                data["success"] = True

    # return json data back to client
    return flask.jsonify(data)
    


if __name__ == "__main__":
    print("[INFO] Loading LSTM model and starting Flask API endpoint.")
    app.run(host='0.0.0.0', port=8000)