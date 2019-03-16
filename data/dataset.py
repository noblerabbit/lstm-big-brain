import io
import numpy as np

class TweetDataset():
    def __init__(self,path,seqlength=40, data_size=0.1):
        self.path = path
        self.char_to_ind = {}
        self.ind_to_char = {}
        self.chars = []
        self.text = ""
        self.sentences = []
        self.next_chars = []
        self.maxlen = seqlength
        
        self.load_source(data_size)
        
    def load_source(self, data_size):
        with io.open(self.path, encoding='utf-8') as f:
            text = f.read()#.lower()
            text = text[:int(len(text)*data_size)] # to fit my gpu
        print('Corpus length: {} charaters.'.format(len(text))
        chars = sorted(list(set(text)))
        print('Total chars:', len(chars))
        self.char_to_ind = {c:i for (i,c) in enumerate(chars)}
        self.ind_to_char = {i:c for (i,c) in enumerate(chars)}
        self.chars = chars
        self.text = text
        
    def create_sequances(self, maxlen = 40):
        step = 3
        # sentences = []
        # next_chars = []
        for i in range(0, len(self.text)-maxlen, step):
            self.sentences.append(self.text[i: i+maxlen])
            self.next_chars.append(self.text[i + maxlen])
        print("Sequences: ", len(self.sentences))
        
    def vectorize_sequances(self):
        print("Sequence Vectroization (one hot encoding)...", end = " ")
        x = np.zeros((len(self.sentences), self.maxlen, len(self.chars)), dtype='bool')
        # print(x.nbytes/10000000)
        y = np.zeros((len(self.sentences), len(self.chars)), dtype='bool')
    
        for i, sentence in enumerate(self.sentences):
            # if i%100000 == 0:
            #     # print(i)
            for t, char in enumerate(sentence):
                x[i, t, self.char_to_ind[char]] = 1
            y[i, self.char_to_ind[self.next_chars[i]]] = 1
        print("DONE!")
        print(x.shape)
        print(y.shape)
        return x, y
        
        
if __name__ == "__main__":
    data = TweetDataset("tumpstweets_source.txt")
    data.create_sequances()
    X, Y = data.vectorize_sequances()
    print(data.chars)