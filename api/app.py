# This is where I can run the trained model and get predictions.
# import keras
import sys
import os
#add root dir to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import flask
from flask import Flask
from data.dataset import TweetDataset
from networks.lstm_basic import lstm_basic
from model.model import LSTMModel
import random
from flask import render_template
import mxnet as mx
from textpredictor import TextPredictor


# init flask
app = Flask(__name__)

path = "static/nietzsche.txt"
params = "static/lstm_basic.params"
pred = TextPredictor(path, params)


@app.route('/')
def index():
    return render_template("one-tweet.html")

@app.route("/predict")
def predict():
    text = pred.predict_text()
    return render_template("one-tweet.html", text=text)


if __name__ == "__main__":
    print("[INFO] Loading LSTM model and starting Flask API endpoint.")
    app.run(host='0.0.0.0')