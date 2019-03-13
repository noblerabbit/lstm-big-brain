# LSTM Big Brain (in progress...)

![Price: free](https://img.shields.io/badge/price-FREE-0098f7.svg)
![Version: 1.0.2](https://img.shields.io/badge/version-1.0.0_-green.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)


**Use this project if you want to train a character predictor. You can deploy the model as Flask API endpoint in the AWS Lambda.**

### Desciption

This project uses Recurrent Neural Network (RNN) with LSTM cells to train the network to randomly generate
tweets. The predictor is trained based on the text file that is located in the "data" directory.
Currently the data directory contains tweets from DJT, however it will train also on any other text file as long as it has more than 200.000 charaters.
The model is build in keras/tensorflow.

In the "api" directory one can run Flask server to test the random tweet generation.

### Quick test

Clone the repo and install dependancies:
```
$ git clone https://github.com/noblerabbit/lstm-big-brain.git
$ cd lstm-big-brain/
$ pip install -r requirements.txt

```

#### 1. Train the model
The model with pretrained weights is located in the "model_data" directory.
You can jump to 2. if you want to test predictions on the trained model immidiately.

**To train the model run "lstm_text_generation.py".**
```
$ python lstm_text_generation.py
```

#### 2. Deploy API endpoint and run predictor
API endpoint runs on Flask framework. The files are located in the "api" directory

**To run the API endpoint:**
```
$ python app.py
```


**To test the inference:**

Open new terminal windows as you need to send post request to API endpoint.
Then use curl to send the request and you will recive a predcition back.

```
$ curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"seed":"seed"}' \
  http://localhost:5000/predict

```


### TODO
- [ ] Optimize for aws lambda
- [ ] Add instuctions how to deploy on aws lambda
- [ ] Create simple UI
- [ ] Train a more sophisticated model for predictions