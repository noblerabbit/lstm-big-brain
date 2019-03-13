# Extract tweet text from json files and saves as one .txt file.

import os
import json

FILENAME = "Tumpstweets_source.txt"
# get filenames
filenames = [filename for filename in os.listdir() if ".json" in filename]
# print(filenames)

# extract tweets source to list
tweets = []
for file in filenames:
    with open(file, 'r') as f:
        data = json.load(f)
        for tweet in data:
            tweets.append(tweet["text"])

    # save list to file
    with open(FILENAME, 'a') as f:
        for tweet in tweets:
            f.write("%s\n" % tweet)
            
    # open file and print sample text from file
    
with open(FILENAME, 'r') as f:
    print(f.readline())
