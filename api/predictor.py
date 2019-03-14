from keras.models import model_from_json
from keras import backend
import numpy as np
import random
import io

class TweetPredictor():
    def __init__(self, model_json, model_weights, vocab_file):
        self.model_json = model_json
        self.model_weights = model_weights
        self.model = self.load_model()
        
        self.text = ""
        self.char_indices = {}
        self.indices_char = {}
        self.chars = []
        self.maxlen = 40
        
        self.load_vocab_(vocab_file)
        # print(type(self.model))
    
    def load_model(self):
        json_file = open(self.model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.model_weights)
        print("Model {} loaded.".format(self.model_json))
        return loaded_model
    
    def sample_(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def load_vocab_(self, vocab_file):
        with io.open(vocab_file, encoding='utf-8') as f:
            self.text = f.read().lower()
            self.text = self.text[:1000000] # to fit my gpu
        print('corpus length:', len(self.text))
        
        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        

    
    def predict_tweet(self, seed=None):
        tweets=[]
        
        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        diversity = [0.2, 0.5, 1.0, 1.2][random.randint(0,4)]
        print('----- diversity:', diversity)
    
        generated = ''
        sentence = self.text[start_index: start_index + self.maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
    
        for i in range(160):
            x_pred = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample_(preds, diversity)
            next_char = self.indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
    
        tweets.append(generated)
        
        return tweets
        
        