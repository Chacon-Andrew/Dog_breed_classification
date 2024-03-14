import os
import train
from flask import Flask, request
import base64

app = Flask(__name__)

@app.route('/guess', methods=['POST'])
def handle_image():
    req = request.data
    string_by = req.decode("utf-8")
    strings = string_by.split(",")
    byteob = strings[1].encode("utf-8")
    decodeit = open("Dog_breed_classification/predict_image.jpeg", 'wb')
    decodeit.write(base64.b64decode(byteob))
    decodeit.close()
    learn = train.learner()
    guess = learn.predict()
    return {"answer": guess}

if __name__ == "__main__":
    app.run(port=8000, debug=True, host="0.0.0.0")