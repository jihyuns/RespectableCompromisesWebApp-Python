# -*- coding: utf-8 -*- 
from gensim.models import Word2Vec
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS             
import os                               
import json

app = Flask(__name__, static_folder='outputs')                 
CORS(app)

def init():
    global model
    model = Word2Vec.load('neighbor_model')   

@app.route("/estimator", methods=['POST'])
def estimator():
    content = request.json
    js = model.most_similar(content['neighbor'])
    result = {'word': js}
    return jsonify(result)

@app.route("/outputs", methods=['GET', 'POST'])
def outputs():
    neighbor_id = request.args.get('neighbor')
    return app.send_static_file(neighbor_id + '.json')

if __name__ == '__main__':
    init()
    app.run('0.0.0.0', port=5000, threaded=True)    # 처리 속도 향상을 위해 쓰레드를 적용한다.

                                                             