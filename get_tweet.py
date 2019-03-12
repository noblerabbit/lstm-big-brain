import requests
import json

headers = {'content-type': 'application/json'}
response = requests.post(url, json={"seed": "obama is the love of my life"}, headers = headers)


"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"seed":"obama is the love of my life"}' \
  http://localhost:5000/predict
  
"""