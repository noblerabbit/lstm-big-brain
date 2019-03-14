# This is where I can run the trained model and get predictions.
# import keras

import flask
from flask import Flask, jsonify

import predictor


path = '../data/tumpstweets_source.txt'
model_file = '../model_data/lstm-big-brain-model_mxnet.json'
weights_file = '../model_data/lstm-big-brain_new_mxnet.h5'

# init flask
app = Flask(__name__)

pred = predictor.TweetPredictor(model_file, weights_file, path)


@app.route('/')
def index():
    return jsonify({"message": "Hello World!"})

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
            # global graph
            # with graph.as_default():
            tweet = pred.predict_tweet()
            data["tweet"] = tweet
            # indicate that the request was a success
            data["success"] = True

    # return json data back to client
    return flask.jsonify(data)
    


if __name__ == "__main__":
    print("[INFO] Loading LSTM model and starting Flask API endpoint.")
    app.run()